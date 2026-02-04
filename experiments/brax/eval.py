"""Efficient evaluation script for the Brax environments.

This script evaluates a trained policy on many tasks and episodes, storing results
in parquet format using polars for fast storage and retrieval.
"""

import logging
import time
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from omegaconf import DictConfig
from tqdm import trange

from rlg.experiments.brax.brax_multi_task_wrapper import TaskParams
from rlg.experiments.brax.cheetah_robust import CheetahRobust
from rlg.experiments.brax.utils import load_env, sample_task
from rlg.experiments.brax.walker_robust import WalkerRobust

logger = logging.getLogger(__name__)


def make_task_sampler(cfg: DictConfig):
    """Creates a task sampler function based on config.

    Distribution: log2(tau) ~ U([log_tau_min, log_tau_max]) for each parameter.
    """
    log_tau_min = cfg.task_sampling.log_tau_min
    log_tau_max = cfg.task_sampling.log_tau_max

    return partial(sample_task, log_tau_min=log_tau_min, log_tau_max=log_tau_max)


def sample_tasks_batch(rng: jax.Array, num_tasks: int, task_sampler) -> TaskParams:
    """Sample a batch of tasks.

    Returns:
        NamedTuple with arrays of shape (num_tasks,) for each parameter.
    """
    keys = jax.random.split(rng, num_tasks)
    tasks = jax.vmap(task_sampler)(keys)
    return tasks


def create_run_batch_episode(
    env: CheetahRobust | WalkerRobust,
    inference_fn,
    episode_length: int,
):
    """Batched episode evaluation."""

    def run_batch_episode(
        rng_batch: jax.Array, task_params_batch: TaskParams
    ) -> jax.Array:
        batch_size = rng_batch.shape[0]

        # Vectorized reset
        split_keys = jax.vmap(jax.random.split)(rng_batch)
        reset_rngs = split_keys[:, 0]
        rng_batch = split_keys[:, 1]
        state_batch = jax.vmap(env.reset)(reset_rngs, task_params_batch)

        def step_fn(carry, _):
            state, rngs, total_reward = carry

            # Split keys per env per step
            split_rngs = jax.vmap(jax.random.split)(rngs)
            act_rngs = split_rngs[:, 0]
            rngs = split_rngs[:, 1]

            action_batch, _ = jax.vmap(inference_fn)(state.obs, act_rngs)
            next_state = jax.vmap(env.step)(state, action_batch)

            total_reward = total_reward + next_state.reward * (1.0 - state.done)
            return (next_state, rngs, total_reward), None

        initial_rewards = jnp.zeros((batch_size,))
        (final_state, final_rngs, final_rewards), _ = jax.lax.scan(
            step_fn,
            (state_batch, rng_batch, initial_rewards),
            None,
            length=episode_length,
        )
        return final_rewards

    return run_batch_episode


def load_policy(cfg: DictConfig, env: CheetahRobust | WalkerRobust, rng: jax.Array):
    """Load the policy network and checkpoint.

    Returns:
        Tuple of (inference_fn, rng) where inference_fn is the JIT-compiled policy.
    """
    ckpt_path = Path(cfg.checkpoint_path)
    logger.info(f"Loading checkpoint from: {ckpt_path.resolve()}")

    # Get observation shape by doing a dummy reset
    rng, init_rng = jax.random.split(rng)
    dummy_task = TaskParams(mass_scale=jnp.array(1.0), length_scale=jnp.array(1.0))
    dummy_state = env.reset(init_rng, task_params=dummy_task)  # type: ignore
    obs_shape = dummy_state.obs.shape  # type: ignore

    # Create policy network
    ppo_network = ppo_networks.make_ppo_networks(
        obs_shape,  # type: ignore
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,  # type: ignore
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Load checkpoint and create inference function
    params = checkpoint.load(ckpt_path.resolve())  # type: ignore
    inference_fn = make_policy(params, deterministic=cfg.eval.deterministic)

    return inference_fn, rng


def sample_and_save_tasks(
    cfg: DictConfig, rng: jax.Array, task_sampler
) -> tuple[TaskParams, jax.Array]:
    """Sample tasks and save their parameters to disk.

    Returns:
        Tuple of (all_tasks, rng) where all_tasks has shape (num_tasks,) per field.
    """
    logger.info(f"Sampling {cfg.eval.num_tasks} tasks...")
    rng, task_rng = jax.random.split(rng)
    all_tasks = sample_tasks_batch(task_rng, cfg.eval.num_tasks, task_sampler)

    # Convert tasks to host memory for storage
    all_tasks_np = jax.tree.map(lambda x: jax.device_get(x), all_tasks)

    # Save task parameters
    tasks_df = pl.DataFrame(
        {
            "task_id": list(range(cfg.eval.num_tasks)),
            "mass_scale": all_tasks_np.mass_scale.tolist(),
            "length_scale": all_tasks_np.length_scale.tolist(),
        }
    )
    tasks_path = Path(cfg.output.dir) / cfg.output.tasks_file
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.write_parquet(tasks_path)
    logger.info(f"Saved task parameters to {tasks_path.resolve()}")

    return all_tasks, rng


def run_evaluation(cfg, rng, all_tasks, run_batch_episode_fn):
    num_tasks = cfg.eval.num_tasks
    num_eps = cfg.eval.num_episodes_per_task
    total_episodes = num_tasks * num_eps

    batch_size = min(cfg.batch_size, total_episodes)

    # Host-side metadata (fine to keep on CPU for output)
    task_indices = np.repeat(np.arange(num_tasks), num_eps)
    episode_indices = np.tile(np.arange(num_eps), num_tasks)

    # JIT the batch runner
    eval_batch_fn = jax.jit(run_batch_episode_fn)

    # --- Compilation Warmup ---
    logger.info(f"Compiling batch evaluation function (batch size={batch_size})...")
    start = time.time()
    dummy_keys = jax.random.split(rng, batch_size)
    dummy_params = TaskParams(
        mass_scale=jnp.ones((batch_size,)), length_scale=jnp.ones((batch_size,))
    )
    compiled = eval_batch_fn.lower(dummy_keys, dummy_params).compile()
    logger.info(f"Compilation complete in {time.time() - start:.2f} seconds")

    logger.info(f"Processing {total_episodes} episodes...")
    start_eval = time.time()

    # Keep results on device until the end
    device_returns = []

    for i in trange(0, total_episodes, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, total_episodes)
        actual_batch_size = end_idx - start_idx

        # Indices for this chunk (host -> device, small)
        current_indices_np = task_indices[start_idx:end_idx]
        current_indices = jnp.asarray(current_indices_np, dtype=jnp.int32)

        # Pad indices ON DEVICE if last chunk is smaller
        if actual_batch_size < batch_size:
            pad_len = batch_size - actual_batch_size
            last = current_indices[-1]
            pad = jnp.full((pad_len,), last, dtype=current_indices.dtype)
            current_indices = jnp.concatenate([current_indices, pad], axis=0)

        # Gather task params ON DEVICE
        chunk_params = TaskParams(
            mass_scale=jnp.take(all_tasks.mass_scale, current_indices),
            length_scale=jnp.take(all_tasks.length_scale, current_indices),
        )

        rng, chunk_key = jax.random.split(rng)
        chunk_keys = jax.random.split(chunk_key, batch_size)

        batch_ret = compiled(chunk_keys, chunk_params)
        device_returns.append(batch_ret)

    # Concatenate on device, then one host transfer
    total_returns = jnp.concatenate(device_returns, axis=0)

    # Drop the padded extras (at most BATCH_SIZE-1)
    total_returns = total_returns[:total_episodes]

    # Ensure timing includes compute
    total_returns.block_until_ready()
    logger.info(f"Evaluation complete in {time.time() - start_eval:.2f} seconds")

    total_returns_np = np.asarray(jax.device_get(total_returns))

    return pl.DataFrame(
        {
            "task_id": task_indices,
            "episode_id": episode_indices,
            "total_return": total_returns_np,
        }
    )


def save_results(cfg: DictConfig, results_df: pl.DataFrame) -> None:
    """Save results and print summary statistics."""
    results_path = Path(cfg.output.dir) / cfg.output.results_file
    results_df.write_parquet(results_path)
    logger.info(f"Saved {len(results_df)} episode results to {results_path}")

    # Print summary statistics
    summary = results_df.group_by("task_id").agg(
        [
            pl.col("total_return").mean().alias("mean_return"),
            pl.col("total_return").std().alias("std_return"),
        ]
    )
    overall_mean = summary["mean_return"].mean()
    overall_std = summary["mean_return"].std()
    logger.info(
        f"Overall mean return across tasks: {overall_mean:.2f} ± {overall_std:.2f}"
    )


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_brax")
def main(cfg: DictConfig):
    """Main evaluation function."""

    logger.info(f"Starting {cfg.env.name} evaluation")

    # Create environment
    env = load_env(cfg.env.name)

    # Initialize RNG
    rng = jax.random.key(cfg.seed)

    # Load policy
    inference_fn, rng = load_policy(cfg, env, rng)

    # Create task sampler and sample tasks
    task_sampler = make_task_sampler(cfg)
    all_tasks, rng = sample_and_save_tasks(cfg, rng, task_sampler)

    # Create vectorized evaluation function
    episode_length = cfg.env.episode_length
    # evaluate_batch = create_vectorized_evaluation(env, inference_fn, episode_length)
    evaluate_batch = create_run_batch_episode(env, inference_fn, episode_length)

    # Run evaluation
    # results_df = run_evaluation(cfg, rng, all_tasks, evaluate_batch)
    results_df = run_evaluation(cfg, rng, all_tasks, evaluate_batch)

    # Save results and print summary
    save_results(cfg, results_df)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

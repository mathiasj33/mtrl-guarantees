"""Efficient evaluation script for the Walker experiment.

This script evaluates a trained policy on many tasks and episodes, storing results
in parquet format using polars for fast storage and retrieval.

Usage:
    python experiments/walker/evaluate.py
    python experiments/walker/evaluate.py eval.num_tasks=1000 eval.num_episodes_per_task=10000
    python experiments/walker/evaluate.py checkpoint.path=/path/to/checkpoint
"""

import logging
import time
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

from rlg.experiments.walker.walker_robust import WalkerRobust, WalkerTaskParams

logger = logging.getLogger(__name__)


def find_latest_checkpoint(base_path: Path) -> Path:
    """Find the latest checkpoint in the given directory."""
    checkpoints = [f for f in base_path.glob("*") if f.is_dir()]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {base_path}")
    return max(checkpoints, key=lambda f: int(f.name))


def get_checkpoint_path(cfg: DictConfig) -> Path:
    """Get the checkpoint path from config or find the latest."""
    if cfg.checkpoint.path is not None:
        ckpt_path = Path(cfg.checkpoint.path)
        if ckpt_path.is_dir():
            subdirs = [f for f in ckpt_path.glob("*") if f.is_dir()]
            if subdirs and all(f.name.isdigit() for f in subdirs):
                return find_latest_checkpoint(ckpt_path)
            return ckpt_path
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")

    # Default: look in experiments/walker/walker_ckp
    default_path = Path(__file__).parent / "walker_ckp"
    if not default_path.exists():
        raise ValueError(
            "No checkpoint path specified and default path not found. "
            "Use checkpoint.path=/path/to/checkpoint"
        )
    return find_latest_checkpoint(default_path)


def make_task_sampler(cfg: DictConfig):
    """Creates a task sampler function based on config.

    Distribution: log2(tau) ~ U([log_tau_min, log_tau_max]) for each parameter.
    """
    log_tau_min = cfg.task_sampling.log_tau_min
    log_tau_max = cfg.task_sampling.log_tau_max

    def sample_task(rng: jax.Array) -> WalkerTaskParams:
        """Samples task parameters from the RoML distribution."""
        log_taus = jax.random.uniform(
            rng, shape=(3,), minval=log_tau_min, maxval=log_tau_max
        )
        taus = 2.0**log_taus

        return WalkerTaskParams(
            mass_scale=taus[0],
            size_scale=taus[1],
            damping_scale=taus[2],
        )

    return sample_task


def sample_tasks_batch(
    rng: jax.Array, num_tasks: int, task_sampler
) -> WalkerTaskParams:
    """Sample a batch of tasks.

    Returns:
        WalkerTaskParams with arrays of shape (num_tasks,) for each parameter.
    """
    keys = jax.random.split(rng, num_tasks)
    tasks = jax.vmap(task_sampler)(keys)
    return tasks


def create_run_single_episode(env: WalkerRobust, inference_fn, episode_length: int):
    """Creates a vectorized episode evaluation function.

    This function efficiently evaluates multiple episodes in parallel on GPU.
    """

    def run_single_episode(rng: jax.Array, task_params: WalkerTaskParams) -> jax.Array:
        """Run a single episode and return total reward.

        Args:
            rng: Random key for this episode
            task_params: Task parameters (scalars, not batched)

        Returns:
            Total reward for the episode
        """
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng, task_params=task_params)

        def step_fn(carry, _):
            state, rng, total_reward = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_rng)
            next_state = env.step(state, action)
            # Accumulate reward, masking after done
            total_reward = total_reward + next_state.reward * (1.0 - state.done)
            return (next_state, rng, total_reward), None

        (_, _, total_reward), _ = jax.lax.scan(
            step_fn,
            (state, rng, jnp.zeros(())),
            None,
            length=episode_length,
        )
        return total_reward

    return run_single_episode


def create_run_batch_episode(
    env: WalkerRobust,
    inference_fn,
    episode_length: int,
    deterministic: bool,
):
    """Batched episode evaluation. If deterministic, avoid per-step RNG splitting."""

    if deterministic:
        fixed_key = jax.random.PRNGKey(0)

        def run_batch_episode(
            rng_batch: jax.Array, task_params_batch: WalkerTaskParams
        ) -> jax.Array:
            batch_size = rng_batch.shape[0]

            # Vectorized reset
            split_keys = jax.vmap(jax.random.split)(rng_batch)
            reset_rngs = split_keys[:, 0]
            state_batch = jax.vmap(env.reset)(reset_rngs, task_params_batch)

            def step_fn(state_and_reward, _):
                state, total_reward = state_and_reward

                # Deterministic inference: pass a fixed key; no splitting
                action_batch, _ = jax.vmap(lambda obs: inference_fn(obs, fixed_key))(
                    state.obs
                )

                next_state = jax.vmap(env.step)(state, action_batch)
                total_reward = total_reward + next_state.reward * (1.0 - state.done)
                return (next_state, total_reward), None

            initial_rewards = jnp.zeros((batch_size,))
            (final_state, final_rewards), _ = jax.lax.scan(
                step_fn,
                (state_batch, initial_rewards),
                None,
                length=episode_length,
            )
            return final_rewards

        return run_batch_episode

    else:

        def run_batch_episode(
            rng_batch: jax.Array, task_params_batch: WalkerTaskParams
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


def load_policy(cfg: DictConfig, env: WalkerRobust, rng: jax.Array):
    """Load the policy network and checkpoint.

    Returns:
        Tuple of (inference_fn, rng) where inference_fn is the JIT-compiled policy.
    """
    ckpt_path = get_checkpoint_path(cfg)
    logger.info(f"Loading checkpoint from: {ckpt_path}")

    # Get observation shape by doing a dummy reset
    rng, init_rng = jax.random.split(rng)
    dummy_task = WalkerTaskParams(
        mass_scale=jnp.array(1.0),
        size_scale=jnp.array(1.0),
        damping_scale=jnp.array(1.0),
    )
    dummy_state = env.reset(init_rng, task_params=dummy_task)
    obs_shape = dummy_state.obs.shape  # type: ignore

    # Create policy network
    ppo_network = ppo_networks.make_ppo_networks(
        obs_shape,  # type: ignore
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,  # type: ignore
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Load checkpoint and create inference function
    params = checkpoint.load(ckpt_path.absolute())  # type: ignore
    inference_fn = make_policy(params, deterministic=cfg.eval.deterministic)

    return inference_fn, rng


def sample_and_save_tasks(
    cfg: DictConfig, rng: jax.Array, task_sampler
) -> tuple[WalkerTaskParams, jax.Array]:
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
            "size_scale": all_tasks_np.size_scale.tolist(),
            "damping_scale": all_tasks_np.damping_scale.tolist(),
        }
    )
    tasks_path = Path(cfg.output.dir) / cfg.output.tasks_file
    tasks_df.write_parquet(tasks_path)
    logger.info(f"Saved task parameters to {tasks_path}")

    return all_tasks, rng


def run_evaluation(cfg, rng, all_tasks, run_batch_episode_fn):
    num_tasks = cfg.eval.num_tasks
    num_eps = cfg.eval.num_episodes_per_task
    total_episodes = num_tasks * num_eps

    BATCH_SIZE = 8192

    # Host-side metadata (fine to keep on CPU for output)
    task_indices = np.repeat(np.arange(num_tasks), num_eps)
    episode_indices = np.tile(np.arange(num_eps), num_tasks)

    # JIT the batch runner
    eval_batch_fn = jax.jit(run_batch_episode_fn)

    # --- Compilation Warmup ---
    logger.info(f"Compiling batch evaluation function (batch size={BATCH_SIZE})...")
    start = time.time()
    dummy_keys = jax.random.split(rng, BATCH_SIZE)
    dummy_params = WalkerTaskParams(
        mass_scale=jnp.ones((BATCH_SIZE,)),
        size_scale=jnp.ones((BATCH_SIZE,)),
        damping_scale=jnp.ones((BATCH_SIZE,)),
    )
    compiled = eval_batch_fn.lower(dummy_keys, dummy_params).compile()
    logger.info(f"Compilation complete in {time.time() - start:.2f} seconds")

    logger.info(f"Processing {total_episodes} episodes...")
    start_eval = time.time()

    # Keep results on device until the end
    device_returns = []

    for i in trange(0, total_episodes, BATCH_SIZE):
        start_idx = i
        end_idx = min(i + BATCH_SIZE, total_episodes)
        actual_batch_size = end_idx - start_idx

        # Indices for this chunk (host -> device, small)
        current_indices_np = task_indices[start_idx:end_idx]
        current_indices = jnp.asarray(current_indices_np, dtype=jnp.int32)

        # Pad indices ON DEVICE if last chunk is smaller
        if actual_batch_size < BATCH_SIZE:
            pad_len = BATCH_SIZE - actual_batch_size
            last = current_indices[-1]
            pad = jnp.full((pad_len,), last, dtype=current_indices.dtype)
            current_indices = jnp.concatenate([current_indices, pad], axis=0)

        # Gather task params ON DEVICE
        chunk_params = WalkerTaskParams(
            mass_scale=jnp.take(all_tasks.mass_scale, current_indices),
            size_scale=jnp.take(all_tasks.size_scale, current_indices),
            damping_scale=jnp.take(all_tasks.damping_scale, current_indices),
        )

        rng, chunk_key = jax.random.split(rng)
        chunk_keys = jax.random.split(chunk_key, BATCH_SIZE)

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


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_walker")
def main(cfg: DictConfig):
    """Main evaluation function."""

    logger.info("Starting Walker evaluation")

    # Create environment
    logger.info(f"Creating WalkerRobust environment with task_mode={cfg.env.task_mode}")
    env = WalkerRobust(task_mode=cfg.env.task_mode)
    # env = BraxAutoResetWrapper(env)

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
    evaluate_batch = create_run_batch_episode(
        env, inference_fn, episode_length, deterministic=cfg.eval.deterministic
    )

    # Run evaluation
    # results_df = run_evaluation(cfg, rng, all_tasks, evaluate_batch)
    results_df = run_evaluation(cfg, rng, all_tasks, evaluate_batch)

    # Save results and print summary
    save_results(cfg, results_df)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

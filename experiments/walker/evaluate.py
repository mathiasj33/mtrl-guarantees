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


def run_evaluation(cfg, rng, all_tasks, run_single_episode_fn):
    num_tasks = cfg.eval.num_tasks  # 1000
    num_eps = cfg.eval.num_episodes_per_task  # 10000
    total_episodes = num_tasks * num_eps

    # SAFE BATCH SIZE: 1 million fits easily on 24GB VRAM
    CHUNK_SIZE = min(1000000, total_episodes)

    # Flatten everything first
    task_indices = np.repeat(np.arange(num_tasks), num_eps)
    episode_indices = np.tile(np.arange(num_eps), num_tasks)

    # Prepare one big compiled function
    # We jit a function that takes a specific number of keys/params (CHUNK_SIZE)
    eval_chunk_fn = jax.jit(jax.vmap(run_single_episode_fn))

    # Compile the function once with dummy data
    logger.info("Compiling evaluation function...")
    start = time.time()
    dummy_keys = jax.random.split(rng, CHUNK_SIZE)
    dummy_params = WalkerTaskParams(
        mass_scale=jnp.ones((CHUNK_SIZE,)),
        size_scale=jnp.ones((CHUNK_SIZE,)),
        damping_scale=jnp.ones((CHUNK_SIZE,)),
    )
    compiled = eval_chunk_fn.lower(dummy_keys, dummy_params).compile()
    logger.info(f"Compilation complete in {time.time() - start:.2f} seconds")

    all_returns = []

    logger.info(f"Processing {total_episodes} episodes in chunks of {CHUNK_SIZE}...")
    start = time.time()

    for i in trange(0, total_episodes, CHUNK_SIZE):
        # Slice indices for this chunk
        current_indices = task_indices[i : i + CHUNK_SIZE]

        # Gather params for this chunk
        # (Assuming all_tasks fields are numpy arrays)
        chunk_mass = all_tasks.mass_scale[current_indices]
        chunk_size = all_tasks.size_scale[current_indices]
        chunk_damp = all_tasks.damping_scale[current_indices]

        chunk_params = WalkerTaskParams(
            mass_scale=jnp.array(chunk_mass),
            size_scale=jnp.array(chunk_size),
            damping_scale=jnp.array(chunk_damp),
        )

        # Generate keys
        rng, chunk_key = jax.random.split(rng)
        chunk_keys = jax.random.split(chunk_key, len(current_indices))

        # Run GPU
        chunk_ret = compiled(chunk_keys, chunk_params)
        all_returns.append(jax.device_get(chunk_ret))

    logger.info(f"Evaluation complete in {time.time() - start:.2f} seconds")

    # Concatenate results
    total_returns_np = np.concatenate(all_returns)

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
    evaluate_single = create_run_single_episode(env, inference_fn, episode_length)

    # Run evaluation
    # results_df = run_evaluation(cfg, rng, all_tasks, evaluate_batch)
    results_df = run_evaluation(cfg, rng, all_tasks, evaluate_single)

    # Save results and print summary
    save_results(cfg, results_df)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

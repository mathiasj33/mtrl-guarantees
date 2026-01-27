"""Efficient evaluation script for the BridgeWorld environment.

This script evaluates deterministic policies (left bridge or right bridge)
on many tasks with different wind parameters, storing results in parquet format
using polars for fast storage and retrieval.
"""

import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import trange

from rlg.environments.bridge_world.bridge_world import (
    BridgeWorld,
    EnvParams,
    EnvState,
)

# Reward thresholds for outcome detection
from rlg.environments.bridge_world.policies import (
    left_bridge_policy,
    right_bridge_policy,
)

logger = logging.getLogger(__name__)


class TaskParams(NamedTuple):
    """Parameters that define a task (wind probabilities)."""

    left_wind_p: jax.Array  # shape: () or (batch,)
    right_wind_p: jax.Array  # shape: () or (batch,)


def sample_task(rng: jax.Array, cfg: DictConfig) -> TaskParams:
    """Sample a single task (wind parameters) based on config.

    Uses uniform distribution for wind probabilities within configured bounds.
    """
    left_key, right_key = jax.random.split(rng)
    left_wind_p = jax.random.uniform(
        left_key, minval=cfg.wind_sampling.left_min, maxval=cfg.wind_sampling.left_max
    )
    right_wind_p = jax.random.uniform(
        right_key,
        minval=cfg.wind_sampling.right_min,
        maxval=cfg.wind_sampling.right_max,
    )
    return TaskParams(left_wind_p=left_wind_p, right_wind_p=right_wind_p)


def sample_tasks_batch(
    rng: jax.Array, num_tasks: int, task_sampler: Callable
) -> TaskParams:
    """Sample a batch of tasks.

    Returns:
        TaskParams with arrays of shape (num_tasks,) for each parameter.
    """
    keys = jax.random.split(rng, num_tasks)
    tasks = jax.vmap(task_sampler)(keys)
    return tasks


def create_run_batch_episode(
    env: BridgeWorld,
    params: EnvParams,
    policy_fn: Callable[[EnvState, EnvParams], jax.Array],
    episode_length: int,
):
    """Create a batched episode runner for the BridgeWorld.

    Args:
        env: The BridgeWorld environment.
        params: Base environment parameters (grid size, bridge positions, etc.).
        policy_fn: Policy function (state, params) -> action.
        episode_length: Maximum number of steps per episode.

    Returns:
        A function that runs batched episodes and returns outcomes.
    """

    def run_batch_episode(
        rng_batch: jax.Array, task_params_batch: TaskParams
    ) -> jax.Array:
        """Run a batch of episodes.

        Args:
            rng_batch: Random keys of shape (batch_size, 2).
            task_params_batch: TaskParams with shape (batch_size,) per field.

        Returns:
            jax.Array: Array of returns with shape (batch_size,).
        """
        batch_size = rng_batch.shape[0]

        # Create initial states with the specified wind parameters
        def make_initial_state(key, left_wind_p, right_wind_p):
            # Use the env's reset to get initial position, but override wind
            base_state = env._reset(key, None, params, None)
            return EnvState(
                position=base_state.position,
                left_wind_p=left_wind_p,
                right_wind_p=right_wind_p,
            )

        split_keys = jax.vmap(jax.random.split)(rng_batch)
        reset_rngs = split_keys[:, 0]
        step_rngs = split_keys[:, 1]

        state_batch = jax.vmap(make_initial_state)(
            reset_rngs,
            task_params_batch.left_wind_p,
            task_params_batch.right_wind_p,
        )

        def step_fn(carry, _):
            state, rngs, done, returns = carry

            # Split keys per env per step
            split_rngs = jax.vmap(jax.random.split)(rngs)
            step_keys = split_rngs[:, 0]
            rngs = split_rngs[:, 1]

            # Get actions from policy
            actions = jax.vmap(policy_fn, in_axes=(0, None))(state, params)

            # Step the environment
            next_state, reward, terminated, _ = jax.vmap(
                env._step, in_axes=(0, 0, 0, None)
            )(step_keys, state, actions, params)

            # Track outcomes (only update if not already done)
            new_returns = returns + jnp.int32((~done) & (reward > 0))
            new_done = done | terminated

            return (next_state, rngs, new_done, new_returns), None

        initial_done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        initial_returns = jnp.zeros((batch_size,), dtype=jnp.int32)

        (_, _, _, final_returns), _ = jax.lax.scan(
            step_fn,
            (state_batch, step_rngs, initial_done, initial_returns),
            None,
            length=episode_length,
        )

        return final_returns

    return run_batch_episode


def get_policy_fn(policy_name: str) -> Callable[[EnvState, EnvParams], jax.Array]:
    """Get the policy function by name."""
    if policy_name == "left_bridge":
        return left_bridge_policy
    elif policy_name == "right_bridge":
        return right_bridge_policy
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def sample_and_save_tasks(
    cfg: DictConfig, rng: jax.Array
) -> tuple[TaskParams, jax.Array]:
    """Sample tasks and save their parameters to disk.

    Returns:
        Tuple of (all_tasks, rng) where all_tasks has shape (num_tasks,) per field.
    """
    logger.info(f"Sampling {cfg.eval.num_tasks} tasks...")
    task_sampler = partial(sample_task, cfg=cfg)
    rng, task_rng = jax.random.split(rng)
    all_tasks = sample_tasks_batch(task_rng, cfg.eval.num_tasks, task_sampler)

    # Convert tasks to host memory for storage
    all_tasks_np = jax.tree.map(lambda x: jax.device_get(x), all_tasks)

    # Save task parameters
    tasks_df = pl.DataFrame(
        {
            "task_id": list(range(cfg.eval.num_tasks)),
            "left_wind_p": all_tasks_np.left_wind_p.tolist(),
            "right_wind_p": all_tasks_np.right_wind_p.tolist(),
        }
    )
    tasks_path = Path(cfg.output.dir) / cfg.output.tasks_file
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.write_parquet(tasks_path)
    logger.info(f"Saved task parameters to {tasks_path.resolve()}")

    return all_tasks, rng


def run_evaluation(
    cfg: DictConfig,
    rng: jax.Array,
    all_tasks: TaskParams,
    run_batch_episode_fn: Callable,
) -> pl.DataFrame:
    """Run the evaluation loop over all tasks and episodes.

    Returns:
        DataFrame with columns: task_id, episode_id, reached_goal, fell_in_abyss, steps
    """
    num_tasks = cfg.eval.num_tasks
    num_eps = cfg.eval.num_episodes_per_task
    total_episodes = num_tasks * num_eps

    batch_size = min(cfg.batch_size, total_episodes)

    # Host-side metadata
    task_indices = np.repeat(np.arange(num_tasks), num_eps)
    episode_indices = np.tile(np.arange(num_eps), num_tasks)

    # JIT compile
    eval_batch_fn = jax.jit(run_batch_episode_fn)

    # Compilation warmup
    logger.info(f"Compiling batch evaluation function (batch size={batch_size})...")
    start = time.time()
    dummy_keys = jax.random.split(rng, batch_size)
    dummy_params = TaskParams(
        left_wind_p=jnp.ones((batch_size,)) * 0.1,
        right_wind_p=jnp.ones((batch_size,)) * 0.1,
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

        # Indices for this chunk
        current_indices_np = task_indices[start_idx:end_idx]
        current_indices = jnp.asarray(current_indices_np, dtype=jnp.int32)

        # Pad indices if last chunk is smaller
        if actual_batch_size < batch_size:
            pad_len = batch_size - actual_batch_size
            last = current_indices[-1]
            pad = jnp.full((pad_len,), last, dtype=current_indices.dtype)
            current_indices = jnp.concatenate([current_indices, pad], axis=0)

        # Gather task params
        chunk_params = TaskParams(
            left_wind_p=jnp.take(all_tasks.left_wind_p, current_indices),
            right_wind_p=jnp.take(all_tasks.right_wind_p, current_indices),
        )

        rng, chunk_key = jax.random.split(rng)
        chunk_keys = jax.random.split(chunk_key, batch_size)

        returns = compiled(chunk_keys, chunk_params)
        device_returns.append(returns)

    # Concatenate on device
    total_returns = jnp.concatenate(device_returns, axis=0)[:total_episodes]

    # Ensure timing includes compute
    total_returns.block_until_ready()
    logger.info(f"Evaluation complete in {time.time() - start_eval:.2f} seconds")

    # Transfer to host
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


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_bridge")
def main(cfg: DictConfig):
    """Main evaluation function."""

    logger.info(f"Starting BridgeWorld evaluation with policy: {cfg.policy}")

    # Create environment
    env = BridgeWorld()
    params = env.default_params

    # Initialize RNG
    rng = jax.random.key(cfg.seed)

    # Get policy function
    policy_fn = get_policy_fn(cfg.policy)

    # Sample tasks and save
    all_tasks, rng = sample_and_save_tasks(cfg, rng)

    # Create evaluation function
    episode_length = cfg.env.episode_length
    run_batch_episode_fn = create_run_batch_episode(
        env, params, policy_fn, episode_length
    )

    # Run evaluation
    results_df = run_evaluation(cfg, rng, all_tasks, run_batch_episode_fn)

    # Save results and print summary
    save_results(cfg, results_df)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

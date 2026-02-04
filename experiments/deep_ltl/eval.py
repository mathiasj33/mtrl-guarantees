"""
Evaluation script for the DeepLTL experiments.

This script evaluates a trained policy on many tasks and episodes, storing results
in parquet format using polars for fast storage and retrieval.
"""

import logging
import time
from pathlib import Path

import equinox as eqx
import hydra
import jax
import numpy as np
import polars as pl
from jaxltl.deep_ltl.eval.eval import Evaluator
from jaxltl.deep_ltl.eval.utils import (
    build_env,
    load_batched_models,
    preprocess_formulas,
)
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from omegaconf import DictConfig
from tqdm import trange

from rlg import MODEL_DIR
from rlg.experiments.deep_ltl.sampler import ReachAvoidFormulaSampler

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_zones")
def main(cfg: DictConfig):
    logger.info(
        f"Running DeepLTL experiment with {cfg.eval.num_tasks} tasks and {cfg.eval.num_episodes_per_task} episodes per task"
    )
    results_df = run(cfg)
    save_results(cfg, results_df)
    logger.info("Evaluation complete!")


def run(cfg: DictConfig) -> pl.DataFrame:
    # build environment
    env, env_params = build_env(cfg, None)

    # sample formulas
    logger.info("Sampling formulas...")
    sampler = ReachAvoidFormulaSampler(
        depth=tuple(cfg.task_sampling.depth),
        reach=tuple(cfg.task_sampling.reach),
        avoid=tuple(cfg.task_sampling.avoid),
        propositions=env.propositions,
    )
    formulas = [sampler.sample() for _ in range(cfg.eval.num_tasks)]

    # save task parameters
    save_tasks(cfg, formulas)

    # construct ldbas and batched sequences for all formulas
    logger.info("Building LDBAs and batched sequences...")
    ldba, batched_seqs = preprocess_formulas(formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    path = MODEL_DIR / "deep_ltl.eqx"
    models = load_batched_models(cfg, path, env, env_params, key=model_key)
    params, static = eqx.partition(models, eqx.is_array)
    params = jax.tree.map(lambda x: x[0], params)  # use first seed
    model = eqx.combine(params, static)

    # run evaluation
    key, eval_key = jax.random.split(key)
    return run_evaluation(cfg, eval_key, model, env, env_params, ldba, batched_seqs)


def save_tasks(cfg: DictConfig, formulas: list[str]) -> None:
    """Save task parameters (formulas) to disk."""
    tasks_df = pl.DataFrame(
        {
            "task_id": list(range(len(formulas))),
            "formula": formulas,
        }
    )
    tasks_path = Path(cfg.output.dir) / cfg.output.tasks_file
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.write_parquet(tasks_path)
    logger.info(f"Saved task parameters to {tasks_path.resolve()}")


def run_evaluation(
    cfg: DictConfig,
    rng: jax.Array,
    model,
    env,
    env_params,
    all_ldbas: JaxLDBA,
    all_batched_seqs: JaxReachAvoidSequence,
) -> pl.DataFrame:
    """Run the evaluation loop over all tasks and episodes.

    Processes tasks in batches of task_batch_size and episodes in batches of
    episode_batch_size. Returns a flat DataFrame with task_id, episode_id, total_return.
    """
    num_tasks = cfg.eval.num_tasks
    num_eps = cfg.eval.num_episodes_per_task
    task_batch_size = cfg.eval.task_batch_size
    episode_batch_size = cfg.eval.episode_batch_size

    assert num_tasks % task_batch_size == 0, (
        f"num_tasks ({num_tasks}) must be divisible by task_batch_size ({task_batch_size})"
    )
    assert num_eps % episode_batch_size == 0, (
        f"num_episodes_per_task ({num_eps}) must be divisible by episode_batch_size ({episode_batch_size})"
    )

    num_task_batches = num_tasks // task_batch_size
    num_episode_batches = num_eps // episode_batch_size

    # Set up evaluator for episode batches
    evaluator = Evaluator(num_episodes=episode_batch_size, discount=0.99)

    # JIT compile the evaluation function
    # vmap over task batch dimension
    eval_fn = eqx.filter_jit(
        eqx.filter_vmap(evaluator.eval, in_axes=(None, None, None, None, 0, 0, 0))
    )

    # Compilation warmup
    logger.info(
        f"Compiling evaluation function (task_batch={task_batch_size}, episode_batch={episode_batch_size})..."
    )
    start = time.time()
    # Get first batch for warmup
    ldba_batch = jax.tree.map(lambda x: x[:task_batch_size], all_ldbas)
    seqs_batch = jax.tree.map(lambda x: x[:task_batch_size], all_batched_seqs)
    warmup_key = jax.random.key(999)
    warmup_keys = jax.random.split(warmup_key, task_batch_size)
    compiled = eval_fn.lower(
        model,
        cfg.eval.deterministic,
        env,
        env_params,
        ldba_batch,
        seqs_batch,
        warmup_keys,
    ).compile()
    logger.info(f"Compilation complete in {time.time() - start:.2f} seconds")

    # Collect results
    batch_results = []

    logger.info(
        f"Processing {num_tasks} tasks x {num_eps} episodes = {num_tasks * num_eps} total episodes..."
    )
    start_eval = time.time()

    for task_batch_idx in trange(num_task_batches, desc="Task batches"):
        task_start = task_batch_idx * task_batch_size
        task_end = task_start + task_batch_size

        # Get LDBA and sequences for this task batch
        def slice_batch(x, start=task_start, end=task_end):
            return x[start:end]

        ldba_batch = jax.tree.map(slice_batch, all_ldbas)
        seqs_batch = jax.tree.map(slice_batch, all_batched_seqs)

        for episode_batch_idx in trange(
            num_episode_batches, desc="     Episode batches", leave=False
        ):
            episode_start = episode_batch_idx * episode_batch_size

            rng, eval_key = jax.random.split(rng)
            task_keys = jax.random.split(eval_key, task_batch_size)
            returns, _, lengths, trajs = compiled(
                model,
                cfg.eval.deterministic,
                env,
                env_params,
                ldba_batch,
                seqs_batch,
                task_keys,
            )  # returns shape: (task_batch_size, episode_batch_size)

            # Convert to host memory and record results
            returns_np = np.asarray(jax.device_get(returns))
            t_ids = np.arange(task_start, task_end).reshape(-1, 1)
            t_ids = np.tile(t_ids, (1, episode_batch_size)).flatten()

            e_ids = np.arange(episode_start, episode_start + episode_batch_size)
            e_ids = np.tile(e_ids, (task_batch_size, 1)).flatten()

            # 5. Create a batch DataFrame (Very fast in Polars)
            batch_df = pl.DataFrame(
                {
                    "task_id": t_ids.astype(np.int32),
                    "episode_id": e_ids.astype(np.int32),
                    "total_return": returns_np.flatten().astype(np.float32),
                }
            )
            batch_results.append(batch_df)

    logger.info(f"Evaluation complete in {time.time() - start_eval:.2f} seconds")

    return pl.concat(batch_results)


def save_results(cfg: DictConfig, results_df: pl.DataFrame) -> None:
    """Save results and print summary statistics."""
    results_path = Path(cfg.output.dir) / cfg.output.results_file
    results_path.parent.mkdir(parents=True, exist_ok=True)
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
        f"Overall mean return across tasks: {overall_mean:.4f} ± {overall_std:.4f}"
    )


if __name__ == "__main__":
    main()

"""Evaluate trained models on a specified set of LTL formulas. Evaluates all models and
formulas in parallel using vmap. Saves results to a CSV file and prints to stdout.

Also supports plotting a subset of trajectories for visual inspection (only for a single formula).
"""

import csv
import logging
import os
import time

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jaxltl.deep_ltl.eval.utils import (
    build_env,
    load_batched_models,
    make_eval_fn,
    preprocess_formulas,
    preprocess_graph_formulas,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    if cfg.plotting.plot and len(cfg.formulas) > 1:
        cfg.plotting.plot = False
        logger.warning(
            "Plotting is only supported for a single formula. Disabling plotting."
        )

    # build environment
    env, env_params = build_env(cfg, None)

    # construct ldbas and batched sequences for all formulas
    # NOTE: consider replacing entirely with preprocess_graph_formulas in future
    if "ltl_gnn" in cfg.model._target_:
        ldba, batched_seqs = preprocess_graph_formulas(cfg.formulas, env)
    else:
        ldba, batched_seqs = preprocess_formulas(cfg.formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models = load_batched_models(cfg, env, env_params, key=model_key)

    # set up evaluator
    eval_fn = make_eval_fn(cfg)

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    metrics, returns, lengths, trajs = eval_fn(
        models,
        cfg.eval.deterministic,
        env,
        env_params,
        ldba,
        batched_seqs,
        eval_key,
    )  # shape: (num_formulas, num_seeds, num_episodes)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    log_and_save_results(cfg, metrics, lengths)

    # plot trajectories
    if cfg.plotting.plot:
        trajs = jax.tree.map(lambda x: x[0, 0, : cfg.plotting.num_trajectories], trajs)
        lengths = jax.tree.map(
            lambda x: x[0, 0, : cfg.plotting.num_trajectories], lengths
        )
        env.plot_trajectories(
            trajs,
            lengths,
            env_params,
            num_cols=cfg.plotting.cols,
            num_rows=cfg.plotting.rows,
        )


def log_and_save_results(cfg: DictConfig, metrics: jax.Array, lengths: jax.Array):
    """Logs aggregated results per formula and saves per-seed results to a CSV file."""

    csv_path = f"runs/{cfg.env.name}/{cfg.run}/eval_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    num_seeds = int(metrics.shape[1])
    seeds = list(range(num_seeds))

    fieldnames = [
        "seed",
        "deterministic",
        "formula",
        "metric",
        "length",
    ]

    rows = []
    for i, formula in enumerate(cfg.formulas):
        # Compute per-seed stats
        metrics_i = metrics[i]  # (num_seeds, num_episodes)
        lengths_i = lengths[i]  # (num_seeds, num_episodes)

        means = jnp.mean(metrics_i, axis=1)  # (num_seeds,)

        success_mask = metrics_i > 0  # (num_seeds, num_episodes)
        success_counts = jnp.sum(success_mask, axis=1)  # (num_seeds,)
        sum_lengths = jnp.sum(lengths_i * success_mask, axis=1)
        avg_lengths = jnp.where(
            success_counts > 0, sum_lengths / success_counts, jnp.nan
        )

        # Stdout logging (aggregate across seeds)
        logger.info("========================================")
        logger.info(f"Formula: {formula}")
        logger.info(f"SR/AV: {float(jnp.mean(means)):.3f}+-{float(jnp.std(means)):.3f}")
        logger.info(
            f"Length: {float(jnp.mean(avg_lengths)):.3f}+-{float(jnp.std(avg_lengths)):.3f}"
        )

        # CSV rows (per-seed)
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "formula": formula,
                    "metric": float(means[seed]),
                    "length": float(avg_lengths[seed]),
                }
            )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote results to {csv_path}")


if __name__ == "__main__":
    main()

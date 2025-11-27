"""Compute evaluation curves by evaluating model checkpoints over time on a fixed set of
formulas. Saves results to a CSV file for later plotting (see scripts/plotting/plot_eval_curves.py).
"""

import csv
import logging
import os
import time

import equinox as eqx
import hydra
import jax
from jax.experimental import io_callback
from omegaconf import DictConfig
from tqdm import tqdm

from jaxltl import DATA_DIR
from jaxltl.deep_ltl.eval.utils import (
    build_env,
    load_model_checkpoints,
    make_eval_fn,
    preprocess_formulas,
    preprocess_graph_formulas,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_curves")
def main(cfg: DictConfig):
    # build environment
    env, env_params = build_env(cfg, None)

    # construct ldbas and batched sequences for all formulas
    logger.info("Processing formulas...")
    formulas_file = DATA_DIR / cfg.env.name / "eval_formulas.txt"
    with open(formulas_file) as f:
        formulas = [line.strip() for line in f.readlines() if line.strip()]
    # NOTE: consider replacing entirely with preprocess_graph_formulas in future
    if "ltl_gnn" in cfg.model._target_:
        ldba, batched_seqs = preprocess_graph_formulas(formulas, env)
    else:
        ldba, batched_seqs = preprocess_formulas(formulas, env)

    logger.info(f"Processed {len(formulas)} formulas.")

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, checkpoint_steps = load_model_checkpoints(
        cfg, env, env_params, key=model_key
    )

    # set up evaluator
    eval_fn = make_eval_fn(cfg)

    # evaluate
    logger.info("Starting evaluation...")
    params, static = eqx.partition(models, eqx.is_array)
    pbar = tqdm(total=len(checkpoint_steps), desc="Evaluating checkpoints", leave=False)

    def update_progress():
        pbar.update(1)  # important: do not return anything

    def eval_timestep(_, model_params):
        model = eqx.combine(model_params, static)
        metrics, returns, lengths, _ = eval_fn(
            model,
            cfg.eval.deterministic,
            env,
            env_params,
            ldba,
            batched_seqs,
            key,
        )  # shape: (num_formulas, num_seeds, num_episodes)
        io_callback(update_progress, None)
        return None, (metrics, returns, lengths)

    start = time.time()
    _, (metrics, returns, lengths) = jax.block_until_ready(
        jax.lax.scan(eval_timestep, None, params)
    )  # shape: (num_checkpoints, num_formulas, num_seeds, num_episodes)
    pbar.close()
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    save_results(cfg, metrics, returns, lengths, checkpoint_steps)


def save_results(
    cfg: DictConfig,
    metrics: jax.Array,
    returns: jax.Array,
    lengths: jax.Array,
    checkpoint_steps: list[int],
):
    """Saves averaged results to a CSV file."""

    csv_path = f"runs/{cfg.env.name}/{cfg.run}/eval_results_checkpoints.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    num_seeds = int(metrics.shape[2])
    seeds = list(range(num_seeds))

    fieldnames = [
        "seed",
        "deterministic",
        "timestep",
        "metric",
        "return",
        "length",
    ]

    rows = []
    for i, step in enumerate(checkpoint_steps):
        # Compute per-seed stats for this timestep
        # shapes: (num_checkpoints, num_formulas, num_seeds, num_episodes)
        mean_metrics, mean_returns, mean_lengths = jax.tree.map(
            lambda x, i=i: x[i].mean(axis=-1).mean(axis=0), (metrics, returns, lengths)
        )

        # CSV rows (per-seed)
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "timestep": step,
                    "metric": float(mean_metrics[seed]),
                    "return": float(mean_returns[seed]),
                    "length": float(mean_lengths[seed]),
                }
            )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote results to {csv_path}")


if __name__ == "__main__":
    main()

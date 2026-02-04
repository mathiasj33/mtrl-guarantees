"""Run beta/delta ablations and generate plots with organized outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import polars as pl


def parse_combos(text: str) -> list[tuple[int, int]]:
    combos = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk:
            left, right = chunk.split("x", 1)
        elif ":" in chunk:
            left, right = chunk.split(":", 1)
        else:
            raise ValueError("Combos must be like '50x100,100x300'.")
        combos.append((int(left), int(right)))
    if not combos:
        raise ValueError("No combos parsed.")
    return combos


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        left, right = [part.strip() for part in chunk.split(",", 1)]
        pairs.append((left, right))
    if not pairs:
        raise ValueError("No beta/delta pairs parsed.")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ZoneEnv")
    parser.add_argument("--run", default="main")
    parser.add_argument(
        "--combos",
        default="50x100,100x300,200x1000",
        help="Comma-separated combos like '50x100,100x300,200x1000'.",
    )
    parser.add_argument(
        "--beta-delta",
        default="1e-4,1e-1;1e-4,1e-2;1e-6,1e-1;1e-6,1e-2;1e-6,1e-3",
        help="Semicolon-separated beta,delta pairs like '1e-4,1e-2;1e-6,1e-3'.",
    )
    args = parser.parse_args()

    combos = parse_combos(args.combos)
    pairs = parse_pairs(args.beta_delta)

    run_root = (Path("runs") / args.env / "eval" / args.run).resolve()
    parquet_path = run_root / "episode_returns.parquet"
    base_guarantees = run_root / "guarantees.csv"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not base_guarantees.exists():
        raise FileNotFoundError(base_guarantees)

    base_df = pd.read_csv(base_guarantees)
    bound = str(base_df["bound"].iloc[0])
    min_return = float(base_df["guarantee"].min())
    max_return = float(base_df["guarantee"].max())

    task_count = pl.read_parquet(parquet_path).select(
        pl.col("task_id").n_unique()
    )[0, 0]

    episode_list = sorted({episodes for _, episodes in combos})
    tasks_list = sorted({tasks for tasks, _ in combos})

    for beta, delta in pairs:
        label = f"beta{beta}_delta{delta}".replace("-", "m")
        out_dir = (run_root / "ablations" / label).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for tasks in tasks_list:
            num_batches = max(1, task_count // tasks)
            cmd = [
                "pixi",
                "run",
                "python",
                "experiments/compute_guarantees_batches.py",
                f"env.name={args.env}",
                f"run={args.run}",
                f"results.dir={out_dir}",
                f"results.source_dir={run_root}",
                "guarantees.use_existing_params=false",
                f"guarantees.param_source_dir={run_root}",
                f"guarantees.bound={bound}",
                f"guarantees.min_return={min_return}",
                f"guarantees.max_return={max_return}",
                f"guarantees.beta={beta}",
                f"guarantees.delta={delta}",
                f"guarantees.batch_size={tasks}",
                f"guarantees.num_batches={num_batches}",
                "guarantees.num_episodes=[{}]".format(
                    ",".join(str(ep) for ep in episode_list)
                ),
            ]
            subprocess.run(cmd, check=True)

        output_dir = (Path("paper/plots") / "ablations" / args.env / label).resolve()
        env = os.environ.copy()
        env["PLOT_RESULTS_DIR"] = str(out_dir)
        env["PLOT_OUTPUT_DIR"] = str(output_dir)
        env["PLOT_COMBOS"] = args.combos
        subprocess.run(["pixi", "run", "python", "paper/plot_zones_avg.py"], check=True, env=env)


if __name__ == "__main__":
    main()

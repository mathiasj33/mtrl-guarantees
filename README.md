# RL Guarantees

This repository contains code for computing and plotting performance guarantees for multi-task reinforcement learning benchmarks. It includes data processing, bound computation, and publication-quality plotting utilities used for the experiments.

## Quick Start

1) Create the environment

```
pixi install
```

2) Compute guarantees for a domain

```
pixi run python experiments/compute_guarantees_batches.py env.name=ZoneEnv run=main
```

3) Plot averaged guarantees

```
pixi run python paper/plot_zones_avg.py
```

## Repository Layout

- `experiments/`: Scripts to compute guarantees and empirical safety curves.
- `paper/`: Plotting scripts and figure outputs used for the paper.
- `conf/`: Hydra configuration files.
- `src/rlg/`: Core library, bounds, and plotting utilities.
- `scripts/`: Helpers for running ablations and generating plots.
- `runs/`: Output data, logs, and CSVs created by experiments.

## Policy Training Notes

- The codebase includes training code using Brax for the MuJoCo benchmarks.
- For DeepLTL, trained policies are provided; they can also be trained using the DeepLTL codebase: https://github.com/...

### Brax training and evaluation

Train a multi-task policy (cheetah or walker):

```
pixi run python experiments/brax/train.py env.name=cheetah run=main
```

Evaluate a checkpoint to produce `episode_returns.parquet`:

```
pixi run python experiments/brax/eval.py env.name=cheetah run=main
```

Optional: render a rollout video from a checkpoint:

```
pixi run python experiments/brax/visualize.py env.name=cheetah run=main
```

All training/evaluation settings are in `conf/train_brax.yaml`, `conf/eval_brax.yaml`, and `conf/visualize_brax.yaml`.

### DeepLTL policies

Pretrained DeepLTL policies are included with the artifacts used for evaluation. To retrain, use the DeepLTL codebase (https://deep-ltl.github.io/).

## Main Experiments

### Compute guarantees

The core experiment computes guarantees for batches of tasks and episodes.

```
pixi run python experiments/compute_guarantees_batches.py env.name=ZoneEnv run=main
```

Key options are controlled through Hydra overrides:

- `guarantees.bound` (e.g., `clopper-pearson`, `hoeffding`, `dkw`, `bernstein`)
- `guarantees.batch_size` (tasks per batch)
- `guarantees.num_batches`
- `guarantees.num_episodes`

### Plot averaged curves

Use the plotting scripts in `paper/` to visualize mean bounds with confidence bands and empirical safety:

```
pixi run python paper/plot_zones_avg.py
pixi run python paper/plot_walker_avg.py
pixi run python paper/plot_cheetah_avg.py
```

These scripts accept optional environment variables for selecting task/episode combinations:

- `PLOT_NUM_TASKS` and `PLOT_NUM_EPISODES` for a single curve
- `PLOT_COMBOS` for multiple curves, e.g. `50x100,100x300,200x1000`

### Bound comparison ablations

To compare bounds (Hoeffding/DKW/Bernstein) at fixed $(\beta,\delta)$:

```
pixi run python scripts/ablate_bounds.py --env ZoneEnv --run main \
  --combos 50x100,100x300,200x1000 \
  --bounds hoeffding,dkw,bernstein \
  --beta 1e-4 --delta 1e-2
```

Plots are saved under:

```
paper/plots/ablations/<env>/bounds_beta<...>_delta<...>/
```

### Beta/Delta ablations

To sweep over $(\beta,\delta)$ combinations and generate organized outputs:

```
pixi run python scripts/ablate_beta_delta.py --env ZoneEnv --run main \
  --combos 50x100,100x300,200x1000 \
  --beta-delta '1e-4,1e-1;1e-4,1e-2;1e-6,1e-1;1e-6,1e-2;1e-6,1e-3'
```

## Reproducibility

- All experiments are driven by Hydra configs in `conf/`.
- Outputs are stored in `runs/<env>/eval/<run>/`.
- Plot scripts are deterministic given the generated CSVs.

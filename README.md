# Probabilistic Performance Guarantees for Multi-Task Reinforcement Learning

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31212/)
[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository contains code for computing and plotting performance guarantees for multi-task reinforcement learning policies. It is the official implementation of
[Probabilistic Performance Guarantees for Multi-Task Reinforcement Learning](https://arxiv.org/pdf/2602.02098).

## Repository Layout

- `experiments/`: Scripts to compute guarantees and evaluate policies.
- `conf/`: Hydra configuration files.
- `src/rlg/`: Core library, bounds, and plotting utilities.
- `data/`: Episode returns and task parameters for each environment (either computed or downloaded from Google Drive, see below).
- `models/`: Pretrained policy checkpoints.
- `guarantees/`: Computed guarantee CSVs.
- `plots/`: Generated figures.
- `dependencies/`: External dependencies (deepltl, mujoco_playground).

## Installation

### 1. Install the environment

This project uses [Pixi](https://pixi.sh/) for dependency management.

```bash
pixi install
```

For GPU support (highly recommended):

```bash
pixi install -e gpu
```

### 2. Install Rabinizer 4 (required for DeepLTL only)

DeepLTL relies on [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html) for the conversion of LTL formulae into LDBAs. Download the program and unzip it into the project directory:

```bash
wget https://www7.in.tum.de/~kretinsk/rabinizer4.zip
unzip rabinizer4.zip
mv rabinizer4 dependencies/jaxltl/dependencies
```

Rabinizer requires Java 11 to be installed on your system and `$JAVA_HOME` to be set accordingly. To test the installation, run:

```bash
./dependencies/jaxltl/dependencies/rabinizer4/bin/ltl2ldba -h
```

This should print a help message.

## Precomputed Experiment Data

Precomputed trajectory returns are available for download:

**Download data:** https://drive.google.com/file/d/1A-fJSuiLhJYlwC4jkzEYNg81lLjCuQrK/view?usp=sharing

Extract the archive into the project root to populate the `data/` directory.

## Computing Guarantees

Compute performance guarantees from episode returns data:

```bash
pixi run python experiments/compute_guarantees.py env.name=cheetah guarantees.bound=bernstein
```

Available environments: `simple_grid`, `bridge_world/left_bridge`, `bridge_world/right_bridge`, `cheetah`, `walker`, `zones`.

Key configuration options (see `conf/compute_guarantees.yaml`):

- `guarantees.bound`: Bound type (`clopper-pearson`, `hoeffding`, `dkw`, `bernstein`)
- `guarantees.num_tasks`: Number of tasks per batch
- `guarantees.num_episodes`: Episodes per task
- `guarantees.num_batches`: Number of independent batches (for computing standard deviations)
- `guarantees.beta`: Per-task confidence level
- `guarantees.delta`: Overall confidence level

## Plotting Guarantees

Generate plots from computed guarantees:

```bash
pixi run python experiments/plot.py env.name=cheetah plot.combinations=[200x1000]
```

Configuration options are in `conf/plot.yaml`. Plots are saved to `plots/<env>/`.

## Evaluating Policies

Pretrained policies are provided in the `models/` directory. To regenerate the episode returns data, or evaluate the policies on a different task distribution:

### Bridge Environment

Evaluate the left or right bridge policy:

```bash
pixi run -e gpu python experiments/bridge_world/eval.py policy=left_bridge
```

Configuration options (see `conf/eval_bridge.yaml`):

- `policy`: `left_bridge` or `right_bridge`
- `wind_sampling`: Slipperiness distribution
- `eval.num_tasks`: Number of tasks to evaluate
- `eval.num_episodes_per_task`: Episodes per task

### Brax Environments (Cheetah, Walker)

Evaluate a trained multi-task policy:

```bash
pixi run -e gpu python experiments/brax/eval.py env.name=cheetah task_sampling.log_tau_min=-1.0 task_sampling.log_tau_max=1.0
```

Configuration options (see `conf/eval_brax.yaml`):

- `eval.num_tasks`: Number of tasks to evaluate
- `eval.num_episodes_per_task`: Episodes per task
- `checkpoint_path`: Path to the model checkpoint

Results are saved to `data/<env>/episode_returns.parquet`.

Optional: render a rollout video:

```bash
pixi run python experiments/brax/visualize.py env.name=cheetah task.mass_scale=1.3 task.length_scale=0.7
```

We provide videos of the pre-trained cheetah and walker policies on various tasks in the `videos/` directory.

### DeepLTL (Zones)

Evaluate the pretrained DeepLTL policy on sampled LTL tasks:

```bash
pixi run -e gpu python experiments/deep_ltl/eval.py
```

Configuration options are in `conf/eval_zones.yaml`:

- `eval.num_tasks`: Number of LTL tasks to sample
- `eval.num_episodes_per_task`: Episodes per task
- `task_sampling.depth/reach/avoid`: Task complexity ranges

Results are saved to `data/zones/episode_returns.parquet`.

## Training Policies

For completeness, we provide code to train multi-task policies from scratch in the environments.

### Brax Environments (Cheetah, Walker)

Train a multi-task policy using PPO:

```bash
pixi run -e gpu python experiments/brax/train.py env.name=cheetah run=tmp
```

Configuration options (see `conf/train_brax.yaml`):

- `env.name`: Environment (`cheetah` or `walker`)
- `task_sampling.log_tau_min/log_tau_max`: Log-uniform task distribution bounds
- `checkpoint_path`: Where to save the trained model

### DeepLTL

Train a DeepLTL policy on the Zones environment using the `jaxltl` dependency:

```bash
cd dependencies/jaxltl
pixi run -e gpu python scripts/train.py experiment=deep_ltl/zones run=tmp num_seeds=1
```

Policies are saved in `dependencies/jaxltl/runs/tmp`.

## License

This project is released under the [Apache License 2.0](LICENSE).

This repository includes the following dependencies with their respective licenses:

- **mujoco_playground**: Apache License 2.0

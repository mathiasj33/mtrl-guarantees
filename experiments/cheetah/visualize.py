"""Visualization script for the robust Walker environment.

This script loads a trained checkpoint and renders a rollout video.

Usage:
    python experiments/walker/visualize.py
    python experiments/walker/visualize.py checkpoint.path=/path/to/checkpoint
    python experiments/walker/visualize.py task.mass_scale=0.5 task.size_scale=2.0
"""

import logging
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import mediapy as media
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from omegaconf import DictConfig
from tqdm import trange

from rlg.experiments.brax.cheetah_robust import CheetahRobust, CheetahTaskParams

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
            # Check if it's a checkpoint directory (contains numbered subdirs)
            subdirs = [f for f in ckpt_path.glob("*") if f.is_dir()]
            if subdirs and all(f.name.isdigit() for f in subdirs):
                return find_latest_checkpoint(ckpt_path)
            return ckpt_path
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")

    default_path = Path("runs/cheetah/cheetah_ckp")
    if not default_path.exists():
        # Try relative to script location
        default_path = Path(__file__).parent / "cheetah_ckp"
    if not default_path.exists():
        raise ValueError(
            "No checkpoint path specified and default path not found. "
            "Use checkpoint.path=/path/to/checkpoint"
        )
    return find_latest_checkpoint(default_path)


@hydra.main(
    version_base="1.1", config_path="../../conf", config_name="visualize_cheetah"
)
def main(cfg: DictConfig):
    """Main visualization function."""
    logger.info("Starting Cheetah visualization")

    # Find checkpoint
    ckpt_path = get_checkpoint_path(cfg)
    logger.info(f"Loading checkpoint from: {ckpt_path}")

    # Create environment
    env = CheetahRobust()

    # Setup task parameters
    task = CheetahTaskParams(
        mass_scale=jnp.array(cfg.task.mass_scale),
        torso_length_scale=jnp.array(cfg.task.length_scale),
    )
    logger.info(
        f"Task parameters: mass={cfg.task.mass_scale}, length={cfg.task.length_scale}"
    )

    # Initialize RNG
    rng = jax.random.key(cfg.seed)

    # JIT compile environment functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Get observation shape
    env_state = jit_reset(rng, task_params=task)
    obs_shape = env_state.obs.shape

    # Create policy network
    ppo_network = ppo_networks.make_ppo_networks(
        obs_shape,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,  # type: ignore
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Load checkpoint and create inference function
    params = checkpoint.load(ckpt_path.absolute())  # type: ignore
    jit_inference_fn = jax.jit(
        make_policy(params, deterministic=cfg.rollout.deterministic)
    )

    # Run rollout
    logger.info(f"Running rollout for {cfg.rollout.num_steps} steps...")
    start = time.time()

    rollout = []
    state = jit_reset(rng, task_params=task)
    rollout.append(state)
    for _ in trange(cfg.rollout.num_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
    logger.info(f"Rollout completed in {time.time() - start:.2f} seconds.")

    # Compute total reward
    rewards = [float(s.reward) for s in rollout]
    total_reward = sum(rewards)
    logger.info(f"Total reward: {total_reward:.2f}")

    # Render video
    render_every = cfg.render.render_every
    output_path = Path(cfg.render.output_path)
    logger.info(f"Rendering video to: {output_path}")

    frames = env.render(task, rollout[::render_every], camera=cfg.render.camera)
    media.write_video(str(output_path), frames, fps=1.0 / env.dt / render_every)

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()

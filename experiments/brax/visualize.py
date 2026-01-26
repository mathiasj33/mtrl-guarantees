"""Visualization script for brax environments.

This script loads a trained checkpoint and renders a rollout video.
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
from rlg.experiments.brax.utils import find_latest_checkpoint
from rlg.experiments.brax.walker_robust import WalkerRobust, WalkerTaskParams

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="visualize_brax")
def main(cfg: DictConfig):
    logger.info(f"Starting {cfg.env.name} visualization")

    # Find checkpoint
    ckpt_path = find_latest_checkpoint(Path(cfg.checkpoint_path))
    logger.info(f"Loading checkpoint from: {ckpt_path}")

    # Create environment
    env, param_clz = {
        "cheetah": (CheetahRobust, CheetahTaskParams),
        "walker": (WalkerRobust, WalkerTaskParams),
    }[cfg.env.name]()

    # Setup task parameters
    task = param_clz(
        mass_scale=jnp.array(cfg.task.mass_scale),
        size_scale=jnp.array(cfg.task.size_scale),
    )
    logger.info(
        f"Task parameters: mass={cfg.task.mass_scale}, size={cfg.task.size_scale}"
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rendering video to: {output_path}")

    frames = env.render(task, rollout[::render_every], camera=cfg.render.camera)
    media.write_video(str(output_path), frames, fps=1.0 / env.dt / render_every)

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()

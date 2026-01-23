"""Training script for the robust Walker environment with multi-task learning.

This script trains a PPO agent on the Walker environment with domain randomization
over physical parameters (mass, head size, damping).

Usage:
    python experiments/walker/train.py
    python experiments/walker/train.py ppo.num_timesteps=50_000_000
    python experiments/walker/train.py env.task_mode=mass
"""

import functools
import logging
from datetime import datetime
from pathlib import Path

import hydra
import jax
from brax.envs.wrappers import training as brax_training
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src.wrapper import BraxAutoResetWrapper
from mujoco_playground.config import dm_control_suite_params
from omegaconf import DictConfig

from rlg.experiments.brax.brax_multi_task_wrapper import BraxMultiTaskWrapper
from rlg.experiments.walker.walker_robust import WalkerRobust, WalkerTaskParams

logger = logging.getLogger(__name__)


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


def make_multi_task_wrapper(task_sampler):
    """Creates an environment wrapper factory for multi-task training."""

    def wrap_env(
        env,
        episode_length: int = 1000,
        action_repeat: int = 1,
        **_kwargs,  # Accept additional kwargs for compatibility with brax
    ):
        wrapped = BraxMultiTaskWrapper(env, task_sampler=task_sampler)
        wrapped = brax_training.VmapWrapper(wrapped)  # type: ignore
        wrapped = brax_training.EpisodeWrapper(wrapped, episode_length, action_repeat)
        wrapped = BraxAutoResetWrapper(wrapped, full_reset=True)
        return wrapped

    return wrap_env


def get_ppo_params() -> dict:
    """Build PPO training parameters from config, using DM Control Suite defaults as base."""
    # Start with DM Control Suite defaults for Walker
    params = dm_control_suite_params.brax_ppo_config("WalkerWalk")
    return dict(params)


def setup_progress_callback():
    """Creates a progress callback that logs training metrics."""
    times = [datetime.now()]
    metrics_history = []

    def progress(num_steps, metrics):
        times.append(datetime.now())
        metrics_history.append({"num_steps": num_steps, **metrics})

        reward = metrics.get("eval/episode_reward", 0)
        reward_std = metrics.get("eval/episode_reward_std", 0)
        logger.info(f"Step {num_steps:,}: reward={reward:.2f} ± {reward_std:.2f}")

    return progress, times, metrics_history


@hydra.main(version_base="1.1", config_path="../../conf", config_name="walker")
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("Starting Walker training")

    # Create environment
    logger.info(f"Creating WalkerRobust environment with task_mode={cfg.env.task_mode}")
    env = WalkerRobust(task_mode=cfg.env.task_mode)

    # Create task sampler
    task_sampler = make_task_sampler(cfg)

    # Build PPO parameters
    ppo_params = get_ppo_params()
    logger.info(f"PPO params: {ppo_params}")

    # Setup network factory
    network_factory = ppo_networks.make_ppo_networks
    ppo_training_params = dict(ppo_params)
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params["network_factory"]
        )

    # Setup progress callback
    progress_fn, times, metrics_history = setup_progress_callback()

    # Setup checkpoint path
    checkpoint_path = Path(cfg.checkpoint.path).absolute()
    logger.info(f"Checkpoints will be saved to: {checkpoint_path}")

    # Create training function
    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress_fn,
        run_evals=True,
        save_checkpoint_path=checkpoint_path,
        seed=cfg.seed,
    )

    # Train
    logger.info("Starting training...")
    start_time = datetime.now()

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=make_multi_task_wrapper(task_sampler),
    )

    end_time = datetime.now()

    # Log timing
    if len(times) > 1:
        jit_time = times[1] - times[0]
        train_time = end_time - times[1]
        logger.info(f"Time to JIT: {jit_time}")
        logger.info(f"Time to train: {train_time}")
    logger.info(f"Total time: {end_time - start_time}")

    # Log final metrics
    if metrics_history:
        final_metrics = metrics_history[-1]
        logger.info(f"Final metrics: {final_metrics}")

    logger.info("Training complete!")
    logger.info(f"Final checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

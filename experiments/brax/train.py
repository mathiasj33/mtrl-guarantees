"""Training script for brax environments with multi-task learning."""

import functools
import logging
from datetime import datetime
from pathlib import Path

import hydra
from brax.envs.wrappers import training as brax_training
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src.wrapper import BraxAutoResetWrapper
from mujoco_playground.config import dm_control_suite_params
from omegaconf import DictConfig

from rlg.experiments.brax.brax_multi_task_wrapper import BraxMultiTaskWrapper
from rlg.experiments.brax.utils import load_env, sample_task

logger = logging.getLogger(__name__)


def make_task_sampler(cfg: DictConfig):
    """Creates a task sampler function based on config.

    Distribution: log2(tau) ~ U([log_tau_min, log_tau_max]) for each parameter.
    """
    log_tau_min = cfg.task_sampling.log_tau_min
    log_tau_max = cfg.task_sampling.log_tau_max

    return functools.partial(
        sample_task, log_tau_min=log_tau_min, log_tau_max=log_tau_max
    )


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


def get_ppo_params(env_name: str) -> dict:
    """Build PPO training parameters from config, using DM Control Suite defaults as base."""
    name_to_dm_control_name = {
        "cheetah": "CheetahRun",
        "walker": "WalkerWalk",
    }
    params = dm_control_suite_params.brax_ppo_config(name_to_dm_control_name[env_name])
    return dict(params)


def setup_progress_callback(total_steps: int):
    """Creates a progress callback that logs training metrics."""
    times = [datetime.now()]
    metrics_history = []

    def progress(num_steps, metrics):
        times.append(datetime.now())
        metrics_history.append({"num_steps": num_steps, **metrics})

        logger.info(f"---- Step {num_steps:,} ----")
        reward = metrics.get("eval/episode_reward", 0)
        reward_std = metrics.get("eval/episode_reward_std", 0)
        logger.info(f"reward={reward:.2f} ± {reward_std:.2f}")
        sps = num_steps / (times[-1] - times[0]).total_seconds()
        logger.info(f"Steps per second: {sps:.2f}")
        remaining = total_steps - num_steps
        if sps > 0:
            eta_seconds = remaining / sps
            logger.info(
                f"Estimated remaining: {eta_seconds // 60}:{eta_seconds % 60:02} (mm:ss)"
            )

    return progress, times, metrics_history


@hydra.main(version_base="1.1", config_path="../../conf", config_name="train_brax")
def main(cfg: DictConfig):
    logger.info(f"Starting training for env={cfg.env.name} with run={cfg.run}")

    # Create environment
    logger.info("Creating environment...")
    env = load_env(cfg.env.name)
    task_sampler = make_task_sampler(cfg)

    # Build PPO parameters
    ppo_params = get_ppo_params(cfg.env.name)
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
    progress_fn, times, metrics_history = setup_progress_callback(
        ppo_params["num_timesteps"]
    )

    # Setup checkpoint path
    checkpoint_path = Path(cfg.checkpoint_path).absolute()
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

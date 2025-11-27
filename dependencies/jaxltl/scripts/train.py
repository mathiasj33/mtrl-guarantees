"""Main training entry point.

Uses Hydra for configuration management. See conf/train.yaml, or run
`python train.py --help` for details.
"""

import datetime
import logging
import time
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import pandas as pd
from omegaconf import DictConfig

import jaxltl
from jaxltl import DATA_DIR, eqx_utils
from jaxltl.deep_ltl.curriculum.curriculum import Curriculum
from jaxltl.deep_ltl.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxltl.environments.spaces import Space
from jaxltl.environments.wrappers import AutoResetWrapper, LogWrapper, VectorizeWrapper
from jaxltl.environments.wrappers.auto_reset_wrapper import ResetStrategy
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.hydra_utils.utils import resolve_default_options
from jaxltl.rl.actor_critic import ActorCritic
from jaxltl.rl.algorithm import RLAlgorithm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if not cfg.use_gpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Using CPU for training")

    default_options = resolve_default_options(cfg.env)

    env, env_params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = f"{DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    if cfg.env.precomputed_curriculum_path is not None:
        precomputed_curriculum_path = (
            f"{DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_curriculum_path}"
        )
    else:
        precomputed_curriculum_path = None
    curriculum: Curriculum = hydra.utils.call(
        cfg.curriculum, load_path=precomputed_curriculum_path
    )
    env = CurriculumWrapper(
        env, curriculum, episode_window=cfg.curriculum_wrapper.episode_window
    )
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )
    env = LogWrapper(env)
    env = VectorizeWrapper(env)

    seeds = jnp.arange(cfg.num_seeds)
    keys = jax.vmap(jax.random.key)(seeds)
    split = jax.vmap(jax.random.split)(keys)
    keys, model_keys = split[:, 0], split[:, 1]

    make_models = eqx.filter_vmap(
        build_model, in_axes=(None, None, None, None, None, 0)
    )
    models = make_models(
        cfg.model,
        env.observation_space(env_params).shape,
        env.action_space(env_params),
        len(env.assignments),
        len(env.propositions),
        model_keys,
    )

    rl_alg: RLAlgorithm = hydra.utils.instantiate(cfg.rl_alg)
    train = eqx.filter_vmap(
        rl_alg.train, in_axes=(eqx.if_array(0), None, None, 0, None, None, 0)
    )
    train = eqx.filter_jit(train)
    logger.info("Compiling training function...")
    start_time = time.time()
    cb = make_callback(cfg)
    compiled = train.lower(
        models, env, env_params, keys, cb, cfg.save_freq, seeds
    ).compile()
    logger.info(f"Compilation completed in {time.time() - start_time:.2f} seconds")

    logger.info("Starting training")
    cb = make_callback(cfg)
    models, metrics = jax.block_until_ready(
        compiled(models, env, env_params, keys, cb, cfg.save_freq, seeds)
    )
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    dfs = []
    for seed in range(cfg.num_seeds):
        seed_metrics = jax.tree.map(lambda x: x[seed], metrics)
        return_values = seed_metrics["episode_return"][seed_metrics["done"]].tolist()
        lengths = seed_metrics["episode_length"][seed_metrics["done"]].tolist()
        stages = seed_metrics["curriculum_stage"][seed_metrics["done"]].tolist()
        timesteps = (
            seed_metrics["total_step"][seed_metrics["done"]] * cfg.rl_alg.num_envs
        ).tolist()
        df = pd.DataFrame(
            {
                "timestep": timesteps,
                "return": return_values,
                "length": lengths,
                "curriculum_stage": stages,
            }
        )
        df["seed"] = seed
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("logs.csv", index=False)

    eqx_utils.save("models.eqx", models, metadata={"num_models": cfg.num_seeds})
    logger.info("Models saved to models.eqx")


def make_callback(cfg: DictConfig):
    """Create a callback function to log progress and save model checkpoints."""

    start_time = time.time()

    def callback(
        metric: dict[str, jax.Array],
        model_params: jax.Array,
        seed: jax.Array,
        step: jax.Array,
    ):
        # estimate remaining training time
        seconds = time.time() - start_time
        sps = step / seconds
        remaining = int((cfg.rl_alg.total_timesteps - step) / sps)
        remaining = str(datetime.timedelta(seconds=remaining))

        # average returns
        window_returns = metric["episode_return"][metric["done"]][
            -cfg.curriculum_wrapper.episode_window :
        ]
        avg_returns = jnp.mean(window_returns)

        window_stages = metric["curriculum_stage"][metric["done"]][
            -cfg.curriculum_wrapper.episode_window :
        ]
        avg_stage = jnp.mean(window_stages)
        min_stage = jnp.min(window_stages)
        max_stage = jnp.max(window_stages)

        # log progress
        logger.info(
            f"seed {seed} | step {step} | ret {avg_returns:.2f} | stage {avg_stage:.2f} ({min_stage:}, {max_stage:}) | sps {int(sps)} | eta {remaining}"
        )

        # save checkpoint
        folder = Path("checkpoints")
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder / f"model_seed{seed}_step{step}.eqx"
        eqx_utils.save(filename, model_params)

    return callback


def build_model(
    model_cfg: DictConfig,
    obs_shape: tuple[int, ...],
    act_space: Space,
    num_assignments: int,
    num_propositions: int,
    key: jax.Array,
) -> ActorCritic:
    model_fn = hydra.utils.instantiate(
        model_cfg,
        obs_shape=obs_shape,
        num_assignments=num_assignments,
        num_propositions=num_propositions,
        key=key,
        _partial_=True,
    )
    # Ensure the space is not converted to an OmegaConf object
    model: ActorCritic = model_fn(act_space=act_space)
    return model


if __name__ == "__main__":
    main()

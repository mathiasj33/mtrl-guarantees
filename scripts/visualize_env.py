"""Script to visualize an environment using a renderer. Supports teleoperation or random actions."""

import hydra
from omegaconf import DictConfig

import jaxltl
from jaxltl.environments.renderer.renderer import BaseRenderer
from jaxltl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.hydra_utils.utils import resolve_default_options


@hydra.main(version_base="1.1", config_path="../conf", config_name="visualize")
def main(cfg: DictConfig):
    default_options = resolve_default_options(cfg.env)

    env, params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        env = PrecomputedResetWrapper(
            env,
            params,
            jaxltl.DATA_DIR / cfg.env.name / cfg.env.precomputed_resets_path,
        )
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )

    renderer: BaseRenderer = env.get_renderer(params)
    renderer.run_render_loop(
        env,
        params,
        policy=cfg.policy,
        print_debug=cfg.print_debug,
    )


if __name__ == "__main__":
    main()

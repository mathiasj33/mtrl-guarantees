"""Script to visualize an environment using a renderer. Supports teleoperation or random actions."""

import hydra
from jaxltl.environments.renderer.renderer import BaseRenderer
from jaxltl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)
from omegaconf import DictConfig

from rlg.environments.bridge_world.bridge_world import BridgeWorld


@hydra.main(version_base="1.1", config_path="../conf", config_name="visualize_bridge")
def main(cfg: DictConfig):
    match cfg.env.name:
        case "BridgeWorld":
            env = BridgeWorld()
            params = env.default_params
        case _:
            raise ValueError(f"Unknown environment: {cfg.env.name}")
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=None
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

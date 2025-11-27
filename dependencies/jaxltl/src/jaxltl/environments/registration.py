from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.letter_world.letter_world import LetterWorld
from jaxltl.environments.rgb_zone_env.rgb_zone_env import RGBZoneEnv
from jaxltl.environments.zone_env.zone_env import ZoneEnv

_name_to_env = {
    "ZoneEnv": ZoneEnv,
    "RGBZoneEnv": RGBZoneEnv,
    "LetterWorld": LetterWorld,
}


def make(name: str, **kwargs) -> tuple[Environment, EnvParams]:
    """Create an environment by name.

    Returns:
        A tuple of the environment instance and its default parameters."""
    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class(**kwargs)
    return env, env.default_params  # type: ignore

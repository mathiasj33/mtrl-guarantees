from collections.abc import Callable

import jax
from mujoco_playground._src.wrapper import Wrapper

from rlg.experiments.walker.walker_robust import WalkerTaskParams


class BraxMultiTaskWrapper(Wrapper):
    """Samples a random task for each episode."""

    def __init__(self, env, task_sampler: Callable[[jax.Array], WalkerTaskParams]):
        """
        Args:
            env: The base environment (WalkerRobust). This should not be vmapped yet.
            task_sampler: Function that samples task parameters given a PRNGKey. Should
                be jittable.
        """
        super().__init__(env)
        self.task_sampler = task_sampler

    def reset(self, rng):
        rng, task_key = jax.random.split(rng)
        task_params = self.task_sampler(task_key)
        return self.env.reset(rng, task_params=task_params)

    def step(self, state, action):
        return self.env.step(state, action)

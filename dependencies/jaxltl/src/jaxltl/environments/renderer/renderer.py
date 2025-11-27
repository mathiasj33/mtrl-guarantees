import sys
from abc import ABC, abstractmethod
from typing import Literal, NamedTuple, override

import jax
import pygame

from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class BaseRenderer[TObsFeatures: NamedTuple, TResetOptions: NamedTuple](ABC):
    """Base class for renderers."""

    def __init__(
        self,
        title: str,
        screen_size: int = 800,
    ):
        self.screen_size = screen_size
        self.title = title

        pygame.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((screen_size, screen_size))

    def get_pressed_keys(self) -> pygame.key.ScancodeWrapper:
        """Gets the currently pressed keys. Exits if QUIT event is detected, or user pressed Q."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit()
        return keys

    def wait_for_keypress(self) -> int:
        """Waits until any key is pressed. Exits if QUIT event is detected, or user pressed Q."""
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                return event.key

    @abstractmethod
    def get_action(self, keys: pygame.key.ScancodeWrapper | int) -> jax.Array:
        """Gets an action from user input."""
        pass

    def close(self):
        """Close the renderer."""
        pygame.quit()

    @abstractmethod
    def run_render_loop(
        self,
        env: Environment | EnvWrapper,
        params: EnvParams,
        policy: Literal["teleop", "random"] = "teleop",
        key: jax.Array | None = None,
        print_debug: bool = False,
    ):
        """Renders the environment in a loop."""
        pass

    def print_obs_and_props(
        self,
        obs: TObsFeatures,
        propositions: jax.Array,
        all_propositions: tuple[str, ...],
    ):
        """Prints the observations and propositions."""
        output = ["\n--- Observations ---\n"]
        output.append(self._format_obs(obs))
        output.append("--- Propositions ---\n")
        output.append(self._format_propositions(propositions, all_propositions))
        output.append("--------------------\n")
        print("".join(output), end="")

    def _format_obs(self, obs: TObsFeatures) -> str:
        """Formats the observations into a string."""
        return ""

    def _format_propositions(
        self, propositions: jax.Array, all_propositions: tuple[str, ...]
    ) -> str:
        """Formats the propositions into a string."""
        lines = []
        true_props = {p for p in propositions.tolist() if p != -1}

        if not all_propositions:
            return ""

        max_len = max(len(p) for p in all_propositions)

        for i, prop_name in enumerate(all_propositions):
            is_true = i in true_props
            lines.append(f"  {prop_name:<{max_len + 1}}: {is_true}\n")
        return "".join(lines)


class ContinuousTimeRenderer[TObsFeatures: NamedTuple, TResetOptions: NamedTuple](
    BaseRenderer[TObsFeatures, TResetOptions]
):
    """Base class for renderers with continuous time."""

    @abstractmethod
    def render(
        self,
        state: WrapperState,
        previous_state: WrapperState,
        obs: TObsFeatures,
        alpha: float,
    ):
        """Renders the environment state. Use alpha for interpolation between frames."""
        pass

    def show_fps(self, clock):
        """Display the current FPS on the window title."""
        fps = clock.get_fps()
        pygame.display.set_caption(f"{self.title} - FPS: {fps:.2f}")

    @override
    def run_render_loop(
        self,
        env: Environment | EnvWrapper,
        params: EnvParams,
        policy: Literal["teleop", "random"] = "teleop",
        key: jax.Array | None = None,
        print_debug: bool = False,
        time_scale: float = 1.0,
    ):
        """Renders the environment in a loop.

        Params:
            env: The environment to render.
            params: Environment parameters.
            policy: Whether to use user input or random actions.
            key: JAX random key. If None, a default key is used.
            print_debug: Whether to print debug information.
            time_scale: Speed multiplier for the simulation.
        """
        if not hasattr(params, "dt"):
            raise ValueError("params must have a 'dt' attribute for time step size.")
        dt = params.dt  # type: ignore

        if key is None:
            key = jax.random.key(0)
        key, key_reset = jax.random.split(key)
        state, obs = env.reset(key_reset, None, params, None)
        action = env.action_space(params).sample(key)  # type: ignore

        # Warm-up step, make sure everything is compiled
        transition = env.step(key, state, action, params)  # type: ignore
        previous_state = state

        clock = pygame.time.Clock()
        time_accumulator = 0.0
        print_debug_time_accumulator = 0.0
        print_debug_interval = 0.1  # seconds

        while True:
            # Get elapsed time in seconds and add to accumulator
            delta_time = (clock.tick(180) / 1000.0) * time_scale
            time_accumulator += delta_time
            self.show_fps(clock)

            # Determine if we should print debug info this frame
            print_debug_this_frame = False
            if print_debug:
                print_debug_time_accumulator += delta_time
                if print_debug_time_accumulator >= print_debug_interval:
                    print_debug_this_frame = True
                    print_debug_time_accumulator -= print_debug_interval

            # Get user action once per frame
            pressed_keys = self.get_pressed_keys()
            key, key_action = jax.random.split(key)  # type: ignore
            action = (
                env.action_space(params).sample(key_action)
                if policy == "random"
                else self.get_action(pressed_keys)
            )

            # Run physics steps to catch up with accumulated time
            while time_accumulator >= dt:
                previous_state = state
                key, key_step = jax.random.split(key)  # type: ignore
                transition = env.step(key_step, state, action, params)
                state = transition.state
                obs = transition.observation

                if transition.reward > 0 and not print_debug:
                    print(f"Reward received: {transition.reward}")

                if transition.truncated or transition.terminated:
                    previous_state = state
                    # If we reset, we can break the inner loop to render the new state
                    break

                time_accumulator -= dt

            # Calculate interpolation factor
            alpha = float(time_accumulator / dt)
            self.render(state, previous_state, obs.features, alpha)

            if print_debug_this_frame:
                self.print_obs_and_props(
                    obs.features, transition.propositions, env.propositions
                )


class DiscreteTimeRenderer[TObsFeatures: NamedTuple, TResetOptions: NamedTuple](
    BaseRenderer[TObsFeatures, TResetOptions]
):
    """Base class for renderers with discrete time. By default, waits for user input
    before each step."""

    @abstractmethod
    def render(self, state: WrapperState, obs: TObsFeatures):
        """Renders the environment state."""
        pass

    @abstractmethod
    def get_action(self, key: int) -> jax.Array:
        """Gets an action from user input."""
        pass

    @override
    def run_render_loop(
        self,
        env: Environment | EnvWrapper,
        params: EnvParams,
        policy: Literal["teleop", "random"] = "teleop",
        key: jax.Array | None = None,
        print_debug: bool = False,
    ):
        """Renders the environment in a loop.

        Params:
            env: The environment to render.
            params: Environment parameters.
            policy: Whether to use user input or random actions.
            key: JAX random key. If None, a default key is used.
            print_debug: Whether to print debug information.
        """
        if key is None:
            key = jax.random.key(0)
        key, key_reset = jax.random.split(key)
        state, obs = env.reset(key_reset, None, params, None)
        action = env.action_space(params).sample(key)  # type: ignore

        # Warm-up step, make sure everything is compiled
        env.step(key, state, action, params)  # type: ignore

        while True:
            self.render(state, obs.features)
            if print_debug:
                propositions = env.compute_propositions(state, params)
                self.print_obs_and_props(obs.features, propositions, env.propositions)
            # Get user action
            pressed_key = self.wait_for_keypress()
            key, key_action = jax.random.split(key)  # type: ignore
            action = (
                env.action_space(params).sample(key_action)
                if policy == "random"
                else self.get_action(pressed_key)
            )
            key, key_step = jax.random.split(key)  # type: ignore
            transition = env.step(key_step, state, action, params)
            state = transition.state
            obs = transition.observation

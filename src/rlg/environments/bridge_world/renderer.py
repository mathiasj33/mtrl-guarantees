from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxltl.environments.renderer.renderer import DiscreteTimeRenderer

from rlg.environments.bridge_world.bridge_world import EnvState


class BridgeWorldRenderer(DiscreteTimeRenderer):
    """Renderer for the BridgeWorld environment."""

    def __init__(
        self,
        title: str,
        screen_size: int = 800,
        grid_size: int = 21,
        bridge_width: int = 3,
        bridge_start_row: int = 5,
        bridge_end_row: int = 15,
    ):
        super().__init__(title=title, screen_size=screen_size)
        self.grid_size = grid_size
        self.bridge_width = bridge_width
        self.bridge_start_row = bridge_start_row
        self.bridge_end_row = bridge_end_row
        self.cell_size = screen_size // self.grid_size

        self._screen_size = self.grid_size * self.cell_size
        if self._screen.get_size() != (self._screen_size, self._screen_size):
            self._screen = pygame.display.set_mode(
                (self._screen_size, self._screen_size)
            )

        # Colors
        self.bg_color = (240, 240, 240)
        self.abyss_color = (64, 64, 64)
        self.bridge_color = (173, 216, 230)  # lightblue
        self.agent_color = (0, 128, 0)  # Green
        self.goal_color = (128, 0, 128)  # Purple
        self.border_color = (0, 0, 0)

        self._canvas = pygame.Surface((self._screen_size, self._screen_size))

    @override
    def render(
        self,
        state: EnvState,
        _,
    ):
        """Renders the environment state."""
        agent_pos = np.array(state.position, dtype=int)
        agent_row, agent_col = agent_pos[0], agent_pos[1]

        self._canvas.fill(self.bg_color)

        bridge_center = self.grid_size // 2
        bridge_half_width = self.bridge_width // 2
        bridge_start_col = bridge_center - bridge_half_width
        bridge_end_col = bridge_center + bridge_half_width

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                bg_to_draw = self.bg_color
                if self.bridge_start_row <= r <= self.bridge_end_row:
                    if bridge_start_col <= c <= bridge_end_col:
                        bg_to_draw = self.bridge_color
                    else:
                        bg_to_draw = self.abyss_color

                if r == 0 and (bridge_start_col <= c <= bridge_end_col):
                    bg_to_draw = self.goal_color

                if (r, c) == (agent_row, agent_col):
                    bg_to_draw = self.agent_color

                pygame.draw.rect(self._canvas, bg_to_draw, rect)
                pygame.draw.rect(self._canvas, self.border_color, rect, 1)

        self._screen.blit(self._canvas, (0, 0))
        pygame.display.flip()

    @override
    def get_action(self, key: int) -> jax.Array:
        """Gets an action from user input."""
        mapping = {
            pygame.K_d: 0,  # right
            pygame.K_s: 1,  # down
            pygame.K_a: 2,  # left
            pygame.K_w: 3,  # up
        }
        if key not in mapping:
            raise ValueError(f"Invalid key pressed: {key}")
        return jnp.array(mapping[key], dtype=jnp.int32)

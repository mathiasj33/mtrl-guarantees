from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxltl.environments.renderer.renderer import DiscreteTimeRenderer

from rlg.environments.bridge_world.bridge_world import EnvState


class BridgeWorldRenderer(DiscreteTimeRenderer):
    """Renderer for the BridgeWorld environment with two bridges."""

    def __init__(
        self,
        title: str,
        screen_size: int = 800,
        grid_width: int = 10,
        grid_height: int = 8,
        bridge_length: int = 4,
        bridge_width: int = 3,
        left_bridge_col: int = 2,
        right_bridge_col: int = 7,
        bridge_start_row: int = 2,
    ):
        super().__init__(title=title, screen_size=screen_size)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.bridge_length = bridge_length
        self.bridge_width = bridge_width
        self.left_bridge_col = left_bridge_col
        self.right_bridge_col = right_bridge_col
        self.bridge_start_row = bridge_start_row
        self.bridge_end_row = bridge_start_row + bridge_length - 1
        self.half_width = bridge_width // 2

        # Calculate cell size to fit the grid
        self.cell_size = min(screen_size // grid_width, screen_size // grid_height)
        self._screen_width = self.grid_width * self.cell_size
        self._screen_height = self.grid_height * self.cell_size

        if self._screen.get_size() != (self._screen_width, self._screen_height):
            self._screen = pygame.display.set_mode(
                (self._screen_width, self._screen_height)
            )

        # Colors
        self.bg_color = (240, 240, 240)
        self.abyss_color = (64, 64, 64)
        self.left_bridge_color = (173, 216, 230)  # lightblue
        self.right_bridge_color = (144, 238, 144)  # lightgreen
        self.agent_color = (255, 165, 0)  # Orange
        self.goal_color = (128, 0, 128)  # Purple
        self.border_color = (0, 0, 0)

        self._canvas = pygame.Surface((self._screen_width, self._screen_height))

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

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                bg_to_draw = self.bg_color

                # Check if in bridge zone (rows where bridges exist)
                if self.bridge_start_row <= r <= self.bridge_end_row:
                    left_start = self.left_bridge_col - self.half_width
                    left_end = self.left_bridge_col + self.half_width
                    right_start = self.right_bridge_col - self.half_width
                    right_end = self.right_bridge_col + self.half_width

                    if left_start <= c <= left_end:
                        bg_to_draw = self.left_bridge_color
                    elif right_start <= c <= right_end:
                        bg_to_draw = self.right_bridge_color
                    else:
                        bg_to_draw = self.abyss_color

                # Goal is only in front of left bridge (top row)
                left_start = self.left_bridge_col - self.half_width
                left_end = self.left_bridge_col + self.half_width
                if r == 0 and left_start <= c <= left_end:
                    bg_to_draw = self.goal_color

                # Agent position
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

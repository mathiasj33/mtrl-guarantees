from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from jaxltl.environments.letter_world.letter_world import EnvState
from jaxltl.environments.renderer.renderer import DiscreteTimeRenderer


class LetterWorldRenderer(DiscreteTimeRenderer):
    """Renderer for the LetterWorld environment."""

    def __init__(
        self,
        title: str,
        screen_size: int = 800,
        grid_size: int = 7,
    ):
        super().__init__(title=title, screen_size=screen_size)
        self.grid_size = grid_size
        self.cell_size = screen_size // self.grid_size
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

        # Ensure proper scaling if the grid isn't perfectly square relative to screen
        self._screen_size = self.grid_size * self.cell_size
        if self._screen.get_size() != (self._screen_size, self._screen_size):
            self._screen = pygame.display.set_mode(
                (self._screen_size, self._screen_size)
            )

        # Colors
        self.bg_color = (240, 240, 240)
        self.cell_color = (200, 200, 200)
        self.agent_color = (0, 128, 0)  # Green
        self.border_color = (0, 0, 0)
        self.text_color_normal = (0, 0, 0)
        self.text_color_agent = (255, 255, 255)

        # Font setup
        self.font = pygame.font.Font(None, int(self.cell_size * 0.75))

        # Pre-create a background surface to clean the screen every frame
        self._canvas = pygame.Surface((self._screen_size, self._screen_size))

    @override
    def render(
        self,
        state: EnvState,
        _,
    ):
        """Renders the environment state."""
        # 1. Extract state data to Numpy for efficient iteration
        # state.letters shape: (G, G, L)
        # state.position shape: (2,)
        grid_data = np.array(state.letters, dtype=int)
        agent_pos = np.array(state.position, dtype=int)

        agent_row, agent_col = agent_pos[0], agent_pos[1]

        # 2. Clear Canvas
        self._canvas.fill(self.bg_color)

        # 3. Iterate over grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # --- Determine Letter ---
                # Find which index is active in the one-hot encoding
                # We assume if max value is roughly 0, it's empty.
                hot_vector = grid_data[r, c]
                max_idx = np.argmax(hot_vector)
                is_active = hot_vector[max_idx] == 1

                current_char = "."
                if is_active and max_idx < len(self.alphabet):
                    current_char = self.alphabet[max_idx]

                # --- Determine Cell Style ---
                bg_to_draw = self.bg_color
                text_color = self.text_color_normal

                # Check if Agent is here
                is_agent_here = (r == agent_row) and (c == agent_col)

                if is_agent_here:
                    bg_to_draw = self.agent_color
                    text_color = self.text_color_agent
                elif current_char != ".":
                    bg_to_draw = self.cell_color

                # --- Draw ---
                # Background
                pygame.draw.rect(self._canvas, bg_to_draw, rect)

                # Text (Letter)
                if current_char != ".":
                    text_surface = self.font.render(current_char, True, text_color)
                    self._canvas.blit(
                        text_surface, text_surface.get_rect(center=rect.center)
                    )

                # Border
                pygame.draw.rect(self._canvas, self.border_color, rect, 2)

        # 4. Blit to screen and update
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

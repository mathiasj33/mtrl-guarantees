"""A 2D renderer for the zone environment based on pygame."""

import math
from functools import partial
from typing import override

import jax
import jax.numpy as jnp
import pygame
from pygame import gfxdraw

from jaxltl.environments.renderer.renderer import ContinuousTimeRenderer
from jaxltl.environments.zone_env.zone_env import (
    EnvParams,
    EnvState,
    ObsFeatures,
    ResetOptions,
)


class Renderer(ContinuousTimeRenderer[ObsFeatures, ResetOptions]):
    def __init__(
        self,
        params: EnvParams,
        screen_size: int = 800,
        grid_size: int = 50,
        show_lidar: bool = False,
    ):
        super().__init__("Zone Environment", screen_size)

        self._params = params
        self._screen_size = screen_size
        self.draw_lidar = show_lidar

        self._background = pygame.Surface(self._screen.get_size())
        self._lidar_surface = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)

        self._world_to_screen_scale = screen_size / params.world_size
        self._agent_radius_px = int(0.1 * self._world_to_screen_scale)
        self._zone_radius_px = int(params.zone_radius * self._world_to_screen_scale)

        # Checkerboard background
        self.grid_size = grid_size
        self._grid_color_1 = (248, 250, 252)
        self._grid_color_2 = (241, 245, 249)
        self._render_background()

        # Agent color
        self._agent_color = (59, 130, 246)  # blue-500
        self._agent_heading_color = (59, 130, 246, 180)  # blue-500 with alpha

        # Color mapping for zones
        self._zone_colors = {
            0: (239, 68, 68, 180),  # red-500
            1: (34, 197, 94, 180),  # green-500
            2: (168, 85, 247, 180),  # purple-500
            3: (234, 179, 8, 180),  # yellow-500
        }

    def _render_background(self):
        """Draw checkerboard background."""
        self._background.fill(self._grid_color_1)
        for y in range(0, self._screen_size, self.grid_size):
            for x in range(0, self._screen_size, self.grid_size):
                if (y // self.grid_size + x // self.grid_size) % 2 == 1:
                    rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
                    self._background.fill(self._grid_color_2, rect)

    def _render_zones(self, state: EnvState):
        centers = self._world_to_screen(state.zone_centers).tolist()
        for i, center in enumerate(centers):
            color_id = int(state.zone_colors[i])
            col = self._zone_colors.get(color_id, (0, 0, 0))
            self._draw_circle(self._screen, col, center, self._zone_radius_px)

    def _draw_circle(self, surface, color, position, radius):
        """Draw an anti-aliased filled circle."""
        gfxdraw.aacircle(surface, position[0], position[1], radius, color)
        gfxdraw.filled_circle(surface, position[0], position[1], radius, color)

    @partial(jax.jit, static_argnames=("self",))
    def _world_to_screen(self, pos: jax.Array) -> jax.Array:
        """Convert world coordinates to screen coordinates."""
        pos = (pos + self._params.world_size / 2) * self._world_to_screen_scale
        pos = pos.at[:, 1].set(self._screen_size - pos[:, 1])
        return pos.astype(jnp.int32)

    @override
    def render(
        self,
        state: EnvState,
        previous_state: EnvState,
        obs: ObsFeatures,
        alpha: float,
    ):
        """Render the environment state."""
        self._screen.blit(self._background, (0, 0))
        self._render_zones(state)

        # Interpolation for smooth rendering
        interpolated_position = (
            previous_state.position * (1.0 - alpha) + state.position * alpha
        )
        angle_diff = (state.angle - previous_state.angle + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        interpolated_angle = previous_state.angle + alpha * angle_diff

        self._draw_agent(interpolated_position, interpolated_angle)
        if self.draw_lidar:
            self._draw_lidar(interpolated_position, obs, state)

        pygame.display.flip()

    @override
    def _format_obs(self, obs: ObsFeatures) -> str:
        """Neatly formats the observations and propositions into a single string."""
        if not isinstance(obs, ObsFeatures):
            return ""

        output = []
        output.append(f"Type: {type(obs).__name__}\n")
        for field, value in obs._asdict().items():
            if not isinstance(value, jax.Array):
                output.append(f"  {field}: {value}\n")
                continue

            if value.ndim == 2:
                output.append(f"  {field}: shape {value.shape}\n")
                if field == "lidar":
                    output.append(self._format_lidar_field(value))
            else:
                with jnp.printoptions(precision=2, suppress=True):
                    output.append(f"  {field}: {value}\n")
        return "".join(output)

    def _format_lidar_field(self, value: jax.Array) -> str:
        lines = []
        num_colors, num_bins = value.shape

        # Header
        header_parts = [f"{'Bin':>3}"]
        header_parts.extend([f"{f'C{i}':>5}" for i in range(num_colors)])
        lines.append(f"    {' | '.join(header_parts)}\n")

        # Separator
        separator_parts = [f"{'-' * 3}"]
        separator_parts.extend([f"{'-' * 5}" for _ in range(num_colors)])
        lines.append(f"    {'-+-'.join(separator_parts)}\n")

        # Data rows
        for i in range(num_bins):
            row_parts = [f"{i:3d}"]
            row_parts.extend([f"{value[j, i]:5.2f}" for j in range(num_colors)])
            lines.append(f"    {' | '.join(row_parts)}\n")

        return "".join(lines)

    def _draw_agent(self, position: jax.Array, angle: jax.Array):
        # Draw agent heading as a rectangle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rect_w = 0.02
        rect_l = 0.2
        corners = jnp.array(
            [
                [0, -rect_w / 2],
                [rect_l, -rect_w / 2],
                [rect_l, rect_w / 2],
                [0, rect_w / 2],
            ]
        )

        # Rotate and translate corners
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = jnp.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + position

        agent_and_corners = jnp.vstack([position, translated_corners])
        screen_positions = self._world_to_screen(agent_and_corners).tolist()
        agent_pos = screen_positions[0]
        self._draw_circle(
            self._screen, self._agent_color, agent_pos, self._agent_radius_px
        )
        corners = screen_positions[1:]
        gfxdraw.filled_polygon(self._screen, corners, self._agent_heading_color)
        gfxdraw.aapolygon(self._screen, corners, self._agent_heading_color)

    def _draw_lidar(
        self,
        position: jax.Array,
        obs: ObsFeatures,
        state: EnvState,
    ):
        self._lidar_surface.fill((0, 0, 0, 0))
        points = self._compute_lidar_points(position, obs, state)  # (C, num_bins, 3)
        screen_pos = self._world_to_screen(points.reshape(-1, 3)[:, :2]).reshape(
            points.shape[0], points.shape[1], 2
        )
        points = points.at[:, :, :2].set(screen_pos).tolist()
        for color, lidar_points in zip(self._zone_colors.values(), points, strict=True):
            for point in lidar_points:
                pos = point[:2]
                strength = point[2]
                if strength > 0.0:
                    exp_strength = math.exp(-2 * (1 - strength))
                    faded_color = (
                        color[0],
                        color[1],
                        color[2],
                        int(exp_strength * 255),
                    )
                    pygame.draw.circle(self._lidar_surface, faded_color, pos, 2)
        self._screen.blit(self._lidar_surface, (0, 0))

    @staticmethod
    @jax.jit
    def _compute_lidar_points(
        position: jax.Array,
        obs: ObsFeatures,
        state: EnvState,
    ) -> jax.Array:
        # obs.lidar shape (C, num_bins)
        num_bins = obs.lidar.shape[1]
        bin_size = 2 * jnp.pi / num_bins
        bin_idx = state.angle // bin_size
        normalized_angle = bin_idx * bin_size
        points = jnp.zeros((*obs.lidar.shape, 3))
        angles = (jnp.arange(num_bins) / num_bins) * 2 * jnp.pi + normalized_angle
        angles = angles % (2 * jnp.pi)
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        for i, signals in enumerate(obs.lidar):
            positions = position + directions * (0.2 + i * 0.1)
            points = points.at[i].set(
                jnp.concatenate([positions, signals[:, None]], axis=-1)
            )
        return points

    @override
    def get_action(self, keys: pygame.key.ScancodeWrapper) -> jax.Array:
        """Gets an action from user input."""

        force = 0.0
        angular_velocity = 0.0

        if keys[pygame.K_w]:
            force = 1.0
        if keys[pygame.K_s]:
            force = -1.0
        if keys[pygame.K_a]:
            angular_velocity = 1.0
        if keys[pygame.K_d]:
            angular_velocity = -1.0

        return jnp.array([force, angular_velocity])

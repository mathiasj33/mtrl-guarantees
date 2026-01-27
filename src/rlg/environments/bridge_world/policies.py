"""Deterministic policies for the BridgeWorld environment.

These policies navigate along the center of each bridge, moving up when aligned
with the target bridge column, and adjusting horizontally otherwise.
When blown left by wind, they simply continue moving up (trying to go straight).
"""

import jax
import jax.numpy as jnp

from .bridge_world import EnvParams, EnvState

# Action mappings (matching bridge_world.py):
# 0: right (+1 column)
# 1: down (+1 row)
# 2: left (-1 column)
# 3: up (-1 row)

ACTION_RIGHT = jnp.int32(0)
ACTION_DOWN = jnp.int32(1)
ACTION_LEFT = jnp.int32(2)
ACTION_UP = jnp.int32(3)


def left_bridge_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Policy that navigates along the center of the left bridge.

    Strategy:
    - If column < left_bridge_col: move right
    - If column > left_bridge_col: move left
    - If aligned with bridge center: move up

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Action to take (integer in [0, 3]).
    """
    row, col = state.position
    bridge_end_row = params.bridge_start_row + params.bridge_length - 1
    reached_bridge = row <= bridge_end_row
    target_col = params.left_bridge_col

    action = jax.lax.cond(
        reached_bridge,
        lambda: ACTION_UP,
        lambda: jax.lax.cond(
            col < target_col,
            lambda: ACTION_RIGHT,
            lambda: jax.lax.cond(
                col > target_col,
                lambda: ACTION_LEFT,
                lambda: ACTION_UP,
            ),
        ),
    )
    return action


def right_bridge_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Policy that navigates along the center of the right bridge.

    Strategy:
    - If column < right_bridge_col: move right
    - If column > right_bridge_col: move left
    - If aligned with bridge center: move up
    - After crossing the bridge (row 0 or 1), move left toward goal

    The goal is above the LEFT bridge, so after crossing the right bridge,
    the agent needs to navigate left to reach the goal.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        Action to take (integer in [0, 3]).
    """
    row = state.position[0]
    col = state.position[1]
    target_col = params.right_bridge_col

    # Determine if we've reached the right bridge
    bridge_end_row = params.bridge_start_row + params.bridge_length - 1
    reached_bridge = row <= bridge_end_row

    # Check if we've crossed the bridge (above bridge rows)
    crossed_bridge = row < params.bridge_start_row

    # Goal region bounds
    half_width = params.bridge_width // 2
    goal_left = params.left_bridge_col - half_width
    goal_right = params.left_bridge_col + half_width

    def navigate_to_goal():
        """After crossing right bridge, navigate to the goal above left bridge."""
        # If we're in the goal region, move up
        in_goal_cols = (col >= goal_left) & (col <= goal_right)
        return jax.lax.cond(
            in_goal_cols,
            lambda: ACTION_UP,
            lambda: ACTION_LEFT,  # Move left toward goal columns
        )

    def navigate_bridge():
        """Navigate up the right bridge."""
        return jax.lax.cond(
            reached_bridge,
            lambda: ACTION_UP,
            lambda: jax.lax.cond(
                col < target_col,
                lambda: ACTION_RIGHT,
                lambda: jax.lax.cond(
                    col > target_col,
                    lambda: ACTION_LEFT,
                    lambda: ACTION_UP,
                ),
            ),
        )

    action = jax.lax.cond(
        crossed_bridge,
        navigate_to_goal,
        navigate_bridge,
    )
    return action


def make_vectorized_policy(policy_fn):
    """Create a vectorized version of a policy for batched evaluation.

    Args:
        policy_fn: A policy function (state, params) -> action.

    Returns:
        A function that takes batched states and params and returns batched actions.
    """
    return jax.vmap(policy_fn, in_axes=(0, None))

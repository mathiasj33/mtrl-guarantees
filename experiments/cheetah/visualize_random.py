"""Visualization script for the robust Cheetah environment.

This script runs a random policy and renders a video for different task instantiations.

Usage:
    python experiments/cheetah/visualize.py
    python experiments/cheetah/visualize.py --mass_scale 0.5 --torso_length_scale 2.0
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy as media
from tqdm import trange

from rlg.experiments.brax.cheetah_robust import CheetahRobust, CheetahTaskParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the robust Cheetah environment with a random policy"
    )
    parser.add_argument(
        "--mass_scale",
        type=float,
        default=1.0,
        help="Mass scaling factor (default: 1.0)",
    )
    parser.add_argument(
        "--torso_length_scale",
        type=float,
        default=0.5,
        help="Torso length scaling factor (default: 1.0)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="videos/cheetah_random.mp4",
        help="Output video path (default: videos/cheetah_random.mp4)",
    )
    parser.add_argument(
        "--render_every",
        type=int,
        default=2,
        help="Render every N steps (default: 2)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="side",
        help="Camera name for rendering (default: side)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Robust Cheetah Environment Visualization")
    print("=" * 60)

    # Create environment
    print("\nCreating CheetahRobust environment...")
    env = CheetahRobust()

    # Setup task parameters
    task = CheetahTaskParams(
        mass_scale=jnp.array(args.mass_scale),
        torso_length_scale=jnp.array(args.torso_length_scale),
    )
    print(
        f"Task parameters: mass_scale={args.mass_scale}, torso_length_scale={args.torso_length_scale}"
    )

    # Initialize RNG
    rng = jax.random.key(args.seed)

    # JIT compile environment functions
    print("\nCompiling environment functions...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Get initial state to determine action size
    state = jit_reset(rng, task_params=task)
    action_size = env.action_size
    print(f"Observation shape: {state.obs.shape}")
    print(f"Action size: {action_size}")

    # Run rollout with random actions
    print(f"\nRunning rollout for {args.num_steps} steps with random policy...")
    start = time.time()

    rollout = []
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng, task_params=task)
    rollout.append(state)

    for _ in trange(args.num_steps):
        rng, action_rng = jax.random.split(rng)
        # Random action in [-1, 1]
        action = jax.random.uniform(action_rng, (action_size,), minval=-1.0, maxval=1.0)
        state = jit_step(state, action)
        rollout.append(state)

    elapsed = time.time() - start
    print(
        f"Rollout completed in {elapsed:.2f} seconds ({args.num_steps / elapsed:.1f} steps/sec)"
    )

    # Compute total reward
    rewards = [float(s.reward) for s in rollout]
    total_reward = sum(rewards)
    print(f"Total reward: {total_reward:.2f}")

    # Render video
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering video to: {output_path}")
    frames = env.render(
        task,
        rollout[:: args.render_every],
        camera=args.camera,
        height=480,
        width=640,
    )

    fps = 1.0 / env.dt / args.render_every
    media.write_video(str(output_path), frames, fps=fps)

    print(f"Video saved! ({len(frames)} frames at {fps:.1f} fps)")
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

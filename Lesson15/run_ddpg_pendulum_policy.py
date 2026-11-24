"""Run Pendulum in human render mode using the DDPG agent trained with Stable-Baselines3.

Run this script from the Lesson15 directory after running
Example_03_DDPG_Pendulum_SB3.ipynb.

Usage:
    python run_ddpg_pendulum_policy.py
    python run_ddpg_pendulum_policy.py --episodes 3
    python run_ddpg_pendulum_policy.py --policy ddpg_pendulum_policy.zip
    python run_ddpg_pendulum_policy.py --delay 0.02

The actor of DDPG is deterministic: the exploration noise is only added
during training, so the agent always takes the action mu(s).

Requires Gymnasium's classic-control renderer:
    pip install "gymnasium[classic-control]"
"""

import argparse
import time

import gymnasium as gym
from stable_baselines3 import DDPG


ENV_MAX_STEPS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="ddpg_pendulum_policy.zip",
        help="path to a saved SB3 DDPG agent",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="extra seconds to wait after each step (Pendulum already renders at 30 fps)",
    )
    args = parser.parse_args()

    # DDPG.load rebuilds the actor and the critic: no network class is needed.
    model = DDPG.load(args.policy, device="cpu")
    print(f"Loaded policy: {args.policy}")

    env = gym.make(
        "Pendulum-v1",
        render_mode="human",
        max_episode_steps=ENV_MAX_STEPS,
    )

    try:
        for episode in range(args.episodes):
            state, info = env.reset()
            episode_return = 0.0

            for step in range(args.max_steps):
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                time.sleep(args.delay)

                # Pendulum never terminates: episodes only end at the time limit.
                if truncated:
                    print(
                        f"Episode {episode + 1}: reached the environment time limit "
                        f"after {step + 1} steps, return={episode_return:.1f}"
                    )
                    break
            else:
                print(
                    f"Episode {episode + 1}: reached the requested "
                    f"{args.max_steps}-step limit, return={episode_return:.1f}"
                )

        time.sleep(1.0)  # Let the last frame remain visible briefly.
    finally:
        env.close()


if __name__ == "__main__":
    main()

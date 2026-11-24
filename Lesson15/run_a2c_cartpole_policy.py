"""Run CartPole in human render mode using the A2C agent trained with Stable-Baselines3.

Run this script from the Lesson15 directory after running
Example_02_A2C_CartPole_SB3.ipynb.

Usage:
    python run_a2c_cartpole_policy.py
    python run_a2c_cartpole_policy.py --episodes 3
    python run_a2c_cartpole_policy.py --policy a2c_cartpole_policy.zip
    python run_a2c_cartpole_policy.py --delay 0.02
    python run_a2c_cartpole_policy.py --stochastic

By default the agent is greedy: it takes the most probable action. With
--stochastic it samples actions from pi(a|s), as during training.

Requires Gymnasium's classic-control renderer:
    pip install "gymnasium[classic-control]"
"""

import argparse
import time

import gymnasium as gym
from stable_baselines3 import A2C


ENV_MAX_STEPS = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="a2c_cartpole_policy.zip",
        help="path to a saved SB3 A2C agent",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="extra seconds to wait after each step (CartPole already renders at 50 fps)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions from the policy instead of choosing the greedy action",
    )
    args = parser.parse_args()

    # A2C.load rebuilds the actor and the critic: no network class is needed.
    model = A2C.load(args.policy, device="cpu")
    print(f"Loaded policy: {args.policy}")
    print(f"Action selection: {'stochastic (sampled)' if args.stochastic else 'greedy'}")

    env = gym.make(
        "CartPole-v1",
        render_mode="human",
        max_episode_steps=ENV_MAX_STEPS,
    )

    try:
        for episode in range(args.episodes):
            state, info = env.reset()
            episode_return = 0.0

            for step in range(args.max_steps):
                action, _ = model.predict(state, deterministic=not args.stochastic)
                state, reward, terminated, truncated, info = env.step(int(action))
                episode_return += reward
                time.sleep(args.delay)

                if terminated:
                    print(
                        f"Episode {episode + 1}: pole fell or cart left bounds "
                        f"after {step + 1} steps, return={episode_return:.0f}"
                    )
                    break

                if truncated:
                    print(
                        f"Episode {episode + 1}: reached the environment time limit "
                        f"after {step + 1} steps, return={episode_return:.0f}"
                    )
                    break
            else:
                print(
                    f"Episode {episode + 1}: reached the requested "
                    f"{args.max_steps}-step limit, return={episode_return:.0f}"
                )

        time.sleep(1.0)  # Let the last frame remain visible briefly.
    finally:
        env.close()


if __name__ == "__main__":
    main()

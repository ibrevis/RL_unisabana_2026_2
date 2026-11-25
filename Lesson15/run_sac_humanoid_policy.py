"""Run Humanoid in human render mode using the SAC agent trained with Stable-Baselines3.

Run this script from the Lesson15 directory after running
Example_04_SAC_Humanoid_SB3.ipynb.

Usage:
    python run_sac_humanoid_policy.py
    python run_sac_humanoid_policy.py --episodes 3
    python run_sac_humanoid_policy.py --policy sac_humanoid_policy.zip
    python run_sac_humanoid_policy.py --delay 0.02
    python run_sac_humanoid_policy.py --stochastic

By default the agent is greedy: it takes the mean of its Gaussian policy.
With --stochastic it samples actions from pi(a|s), as during training.

Requires Gymnasium's MuJoCo environments:
    pip install "gymnasium[mujoco]"
"""

import argparse
import time

import gymnasium as gym
from stable_baselines3 import SAC


ENV_MAX_STEPS = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="sac_humanoid_policy.zip",
        help="path to a saved SB3 SAC agent",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="extra seconds to wait after each step",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions from the policy instead of taking its mean",
    )
    args = parser.parse_args()

    # SAC.load rebuilds the actor and the critics: no network class is needed.
    model = SAC.load(args.policy, device="cpu")
    print(f"Loaded policy: {args.policy}")
    print(f"Action selection: {'stochastic (sampled)' if args.stochastic else 'greedy (mean)'}")

    env = gym.make(
        "Humanoid-v5",
        render_mode="human",
        max_episode_steps=ENV_MAX_STEPS,
        default_camera_config={
        "type": 1,          # tracking camera: follow a body
        "trackbodyid": 1,   # body 1 is the torso
        "distance": 4.0,
        "elevation": -20.0,
    },
    )

    try:
        for episode in range(args.episodes):
            state, info = env.reset()
            episode_return = 0.0

            for step in range(args.max_steps):
                action, _ = model.predict(state, deterministic=not args.stochastic)
                state, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                time.sleep(args.delay)

                if terminated:
                    print(
                        f"Episode {episode + 1}: the robot fell (torso left the healthy "
                        f"height range) after {step + 1} steps, return={episode_return:.0f}"
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

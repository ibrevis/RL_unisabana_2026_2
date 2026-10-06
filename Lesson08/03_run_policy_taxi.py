"""Run Taxi in human render mode using a policy learned in the notebook.

Usage:
    python 03_run_policy_taxi.py                          # 1 episode, default policy_taxi.npy
    python 03_run_policy_taxi.py --episodes 3             # 3 episodes back to back
    python 03_run_policy_taxi.py --q other_policy.npy     # a different saved table

Requires the human renderer:  pip install "gymnasium[toy-text]"
"""

import argparse
import time

import gymnasium as gym
import numpy as np

ACTION_NAMES = ["south", "north", "east", "west", "pickup", "dropoff"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default="policy_taxi.npy", help="path to the saved policy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between steps")
    args = parser.parse_args()

    q_table = np.load(args.q)
    print(f"Loaded {args.q} with shape {q_table.shape}")

    env = gym.make("Taxi-v4", render_mode="human")

    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(args.max_steps):
            # Greedy w.r.t. the learned values -- no epsilon, no exploration.
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(args.delay)

            if terminated or truncated:
                outcome = "delivered the passenger" if terminated else "timed out"
                print(f"Episode {episode + 1}: {outcome} "
                      f"after {step + 1} steps, reward={total_reward}")
                break
        else:
            print(f"Episode {episode + 1}: hit the {args.max_steps}-step limit")

    time.sleep(1.0)  # let the last frame stay on screen
    env.close()


if __name__ == "__main__":
    main()

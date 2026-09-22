"""Run FrozenLake in human render mode using a policy learned in the notebook.

Usage:
    python run_policy_4x4.py                      # 1 episode, default policy.npy
    python run_policy_4x4.py --episodes 3         # 3 episodes back to back
    python run_policy_4x4.py --q other_policy.npy  # a different saved table

Requires the human renderer:  pip install "gymnasium[toy-text]"
"""

import argparse
import time

import gymnasium as gym
import numpy as np

# These MUST match the env the q_table was trained on. A policy learned with
# is_slippery=False has never had to hedge against slipping, so replaying it on
# a slippery env will walk it straight into a hole.
ENV_KWARGS = dict(
    map_name="4x4",
    is_slippery=False,
    reward_schedule=(1, -1, 0),  # (goal, hole, step)
)

ACTION_NAMES = ["left", "down", "right", "up"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default="policy.npy", help="path to the saved policy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between steps")
    args = parser.parse_args()

    q_table = np.load(args.q)
    print(f"Loaded {args.q} with shape {q_table.shape}")

    env = gym.make("FrozenLake-v1", render_mode="human", **ENV_KWARGS)

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
                outcome = "reached the goal" if reward > 0 else "fell in / timed out"
                print(f"Episode {episode + 1}: {outcome} "
                      f"after {step + 1} steps, reward={total_reward}")
                break
        else:
            print(f"Episode {episode + 1}: hit the {args.max_steps}-step limit")

    time.sleep(1.0)  # let the last frame stay on screen
    env.close()


if __name__ == "__main__":
    main()

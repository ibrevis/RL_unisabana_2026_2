"""Run CartPole in human render mode using a policy learned with REINFORCE.

Run this script from the cart_pole_example directory after running one of the
REINFORCE notebooks.

Usage:
    python 06_run_reinforce_cartpole_policy.py
    python 06_run_reinforce_cartpole_policy.py --episodes 3
    python 06_run_reinforce_cartpole_policy.py --policy reinforce_cartpole_policy.pth
    python 06_run_reinforce_cartpole_policy.py --delay 0.05

Requires Gymnasium's classic-control renderer:
    pip install "gymnasium[classic-control]"
"""

import argparse
import time

import gymnasium as gym
import torch
import torch.nn as nn


STATE_SIZE = 4
ACTION_SIZE = 2
HIDDEN_SIZE = 128
ENV_MAX_STEPS = 500


class PolicyNetwork(nn.Module):
    """Map a CartPole observation to one logit per action."""

    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.net(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="reinforce_cartpole_policy.pth",
        help="path to a saved REINFORCE policy",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="seconds to wait after each step",
    )
    args = parser.parse_args()

    # Recreate the network used during training, then load its saved weights.
    policy_net = PolicyNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    saved_weights = torch.load(
        args.policy,
        map_location="cpu",
        weights_only=True,
    )
    policy_net.load_state_dict(saved_weights)
    policy_net.eval()
    print(f"Loaded policy: {args.policy}")

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
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                ).unsqueeze(0)

                # Greedy action: choose the action with the largest policy logit.
                with torch.no_grad():
                    logits = policy_net(state_tensor)
                    action = int(logits.argmax(dim=1).item())

                state, reward, terminated, truncated, info = env.step(action)
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

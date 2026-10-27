"""Run LunarLander in human render mode using the DQN policy learned in the notebook.

Usage:
    python run_dqn_lunarlander_policy.py
    python run_dqn_lunarlander_policy.py --episodes 3
    python run_dqn_lunarlander_policy.py --policy another_policy.pth
    python run_dqn_lunarlander_policy.py --delay 0.02

Requires Box2D (it may need `pip install swig` first):
    pip install "gymnasium[box2d]"
"""

import argparse
import time

import gymnasium as gym
import torch
import torch.nn as nn


STATE_SIZE = 8
ACTION_SIZE = 4
HIDDEN_SIZE = 128
ENV_MAX_STEPS = 1000


class QNetwork(nn.Module):
    """Map a LunarLander observation to one Q-value per action."""

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
        default="dqn_lunarlander_policy.pth",
        help="path to the saved DQN policy",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to wait after each step",
    )
    args = parser.parse_args()

    # Recreate the network used during training, then load its saved weights.
    q_net = QNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    saved_weights = torch.load(
        args.policy,
        map_location="cpu",
        weights_only=True,
    )
    q_net.load_state_dict(saved_weights)
    q_net.eval()
    print(f"Loaded policy: {args.policy}")

    env = gym.make(
        "LunarLander-v3",
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

                # Greedy action: no epsilon and no exploration during evaluation.
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = int(q_values.argmax(dim=1).item())

                state, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                time.sleep(args.delay)

                if terminated:
                    print(
                        f"Episode {episode + 1}: landed, crashed or flew away "
                        f"after {step + 1} steps, return={episode_return:.1f}"
                    )
                    break

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

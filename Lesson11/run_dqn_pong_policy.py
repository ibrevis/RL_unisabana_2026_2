"""Run Atari Pong in human render mode using the DQN policy learned in the notebook.

Usage:
    python run_dqn_pong_policy.py
    python run_dqn_pong_policy.py --episodes 3
    python run_dqn_pong_policy.py --policy dqn_pong_policy_best.pth
    python run_dqn_pong_policy.py --epsilon 0.0

Requires the Atari environments and OpenCV:
    pip install "gymnasium[atari]" opencv-python
"""

import argparse
import random

import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class QNetwork(nn.Module):
    """Map 4 stacked 84x84 frames to one Q-value per action."""

    def __init__(self, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, state):
        # uint8 pixels in [0, 255] -> floats in [0, 1]
        return self.net(state.float() / 255.0)


def make_env():
    """Same preprocessing as in the notebook, but rendered on the screen."""
    env = gym.make(
        "ALE/Pong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="human",
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="dqn_pong_policy.pth",
        help="path to the saved DQN policy",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="probability of a random action (0.05 as in the evaluation)",
    )
    args = parser.parse_args()

    gym.register_envs(ale_py)
    env = make_env()

    # Recreate the network used during training, then load its saved weights.
    q_net = QNetwork(env.action_space.n)
    saved_weights = torch.load(
        args.policy,
        map_location="cpu",
        weights_only=True,
    )
    q_net.load_state_dict(saved_weights)
    q_net.eval()
    print(f"Loaded policy: {args.policy}")

    try:
        for episode in range(args.episodes):
            state, info = env.reset()
            agent_points = 0
            opponent_points = 0
            done = False

            while not done:
                if random.random() < args.epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.as_tensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = q_net(state_tensor)
                        action = int(q_values.argmax(dim=1).item())

                state, reward, terminated, truncated, info = env.step(action)
                agent_points += reward > 0
                opponent_points += reward < 0
                done = terminated or truncated

            print(f"Episode {episode + 1}: agent {agent_points} - {opponent_points} opponent")
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Train DQN on Atari Pong (same code as 03_DQN_Pong_sol.ipynb), for a GPU server.

Setup on the server:
    pip install torch "gymnasium[atari]>=1.0" opencv-python matplotlib numpy
    (the Atari ROMs are included in the ale-py package)

Run in the background and follow the log:
    nohup python 03_DQN_Pong_train.py > pong_train.log 2>&1 &
    tail -f pong_train.log

Quick test with tiny settings (a few seconds):
    python 03_DQN_Pong_train.py --smoke

Output files (in the current directory):
    dqn_pong_policy.pth           last network (saved every CHECKPOINT_EVERY episodes and at the end)
    dqn_pong_policy_best.pth      network with the best 20-episode mean return
    dqn_pong_returns.npy          training return of every episode
    dqn_pong_learning_curve.png   learning curve

Copy dqn_pong_policy.pth and dqn_pong_returns.npy next to 03_DQN_Pong_sol.ipynb
and run the notebook with TRAIN = False.
"""

import argparse
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

matplotlib.use("Agg")  # no screen on the server: save the figure to a file
from matplotlib import pyplot as plt  # noqa: E402


# Training configuration
NUM_EPISODES = 800
MAX_STEPS_PER_EPISODE = 10000
GAMMA = 0.99
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
BUFFER_SIZE = 10000
MIN_BUFFER = 10000
TRAIN_FREQ = 1                    # one gradient step every TRAIN_FREQ environment steps
TARGET_UPDATE_STEPS = 1000        # copy the online network every TARGET_UPDATE_STEPS steps
GRAD_CLIP = 10.0

# Epsilon-greedy exploration schedule (linear in the number of environment steps)
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_STEPS = 150000

STOP_AT_MEAN = 19.0         # stop when the mean of the last 20 training returns reaches this value
EVAL_EPISODES = 10
EVAL_EPSILON = 0.05

SEED = 42

CHECKPOINT_EVERY = 25       # episodes
MODEL_PATH = "dqn_pong_policy.pth"
BEST_MODEL_PATH = "dqn_pong_policy_best.pth"
RETURNS_PATH = "dqn_pong_returns.npy"
CURVE_PATH = "dqn_pong_learning_curve.png"


def make_env():
    """Create Pong with the preprocessing of the DQN papers."""
    env = gym.make(
        "ALE/Pong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
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


def epsilon_by_step(step):
    """Linear decay from EPSILON_MAX to EPSILON_MIN, then constant."""
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_MAX + fraction * (EPSILON_MIN - EPSILON_MAX)


def epsilon_greedy(env, q_net, state, epsilon, device):
    """Choose a random action or the action with the largest Q-value."""
    if random.random() < epsilon:
        return env.action_space.sample()

    # The state stays uint8; the network converts it to floats
    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = q_net(state_tensor)

    return int(q_values.argmax(dim=1).item())


def train_step(q_net, target_net, optimizer, loss_fn, replay_buffer, device):
    """Perform one DQN update from a random replay minibatch."""
    if len(replay_buffer) < MIN_BUFFER:
        return None

    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, terminated = zip(*batch)

    states = torch.as_tensor(np.array(states), device=device)
    actions = torch.as_tensor(
        actions, dtype=torch.int64, device=device
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        rewards, dtype=torch.float32, device=device
    ).unsqueeze(1)
    next_states = torch.as_tensor(np.array(next_states), device=device)
    terminated = torch.as_tensor(
        terminated, dtype=torch.float32, device=device
    ).unsqueeze(1)

    all_q_values = q_net(states)
    selected_q_values = all_q_values.gather(1, actions)

    with torch.no_grad():
        all_next_q_values = target_net(next_states)
        max_next_q_values = all_next_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards + GAMMA * max_next_q_values * (1.0 - terminated)

    loss = loss_fn(selected_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return loss.item()


def save_weights(q_net, path):
    """Save CPU copies of the weights, so the file loads on any machine."""
    torch.save({name: w.cpu() for name, w in q_net.state_dict().items()}, path)


def save_learning_curve(rewards_all_episodes, path):
    window = 20
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_all_episodes, alpha=0.3, label="Return per episode")
    if len(rewards_all_episodes) >= window:
        rolling_rewards = np.convolve(
            np.array(rewards_all_episodes),
            np.ones(window) / window,
            mode="valid",
        )
        plt.plot(np.arange(window - 1, len(rewards_all_episodes)), rolling_rewards, label=f"Rolling average ({window} episodes)")
    plt.axhline(STOP_AT_MEAN, color="tab:red", linestyle="--", label="Stop threshold")
    plt.xlabel("Episode")
    plt.ylabel("Return (points won - points lost)")
    plt.ylim(-22, 22)
    plt.title(f"DQN on Pong: rolling average return (window = {window} episodes)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    global NUM_EPISODES, BUFFER_SIZE, MIN_BUFFER, EVAL_EPISODES, CHECKPOINT_EVERY

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    args = parser.parse_args()
    if args.smoke:
        NUM_EPISODES = 2
        BUFFER_SIZE = 500
        MIN_BUFFER = 500
        EVAL_EPISODES = 1
        CHECKPOINT_EVERY = 1
        print("SMOKE TEST: tiny run, the results are meaningless", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    gym.register_envs(ale_py)
    env = make_env()
    env.action_space.seed(SEED)
    action_size = env.action_space.n

    print("State shape:", env.observation_space.shape, env.observation_space.dtype, flush=True)
    print("Action size:", action_size, flush=True)
    print("Device:", device, flush=True)

    # Online network: updated by gradient descent
    q_net = QNetwork(action_size).to(device)

    # Target network: supplies stable Bellman targets
    target_net = QNetwork(action_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    # ------------------------------------------------------------------ training
    rewards_all_episodes = []
    losses = []
    total_steps = 0
    best_mean = -np.inf

    start_time = time.time()

    for episode in range(NUM_EPISODES):
        if episode == 0:
            state, info = env.reset(seed=SEED)
        else:
            state, info = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            epsilon = epsilon_by_step(total_steps)
            action = epsilon_greedy(env, q_net, state, epsilon, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # No .copy(): the wrapper returns a new array at every step
            replay_buffer.append((
                state,
                action,
                reward,
                next_state,
                float(terminated),
            ))

            if total_steps % TRAIN_FREQ == 0:
                loss = train_step(q_net, target_net, optimizer, loss_fn, replay_buffer, device)
                if loss is not None:
                    losses.append(loss)

            if total_steps % TARGET_UPDATE_STEPS == 0:
                target_net.load_state_dict(q_net.state_dict())

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        rewards_all_episodes.append(episode_reward)
        recent_mean = np.mean(rewards_all_episodes[-20:])

        # Keep the network with the best 20-episode mean (after at least 20 episodes)
        if len(rewards_all_episodes) >= 20 and recent_mean > best_mean:
            best_mean = recent_mean
            save_weights(q_net, BEST_MODEL_PATH)

        if episode == 0 or (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Episode {episode + 1:3d}: "
                f"reward={episode_reward:5.0f}, "
                f"mean20={recent_mean:6.1f}, "
                f"epsilon={epsilon:.3f}, "
                f"steps={total_steps}, "
                f"speed={total_steps / elapsed:.0f} steps/s, "
                f"time={elapsed / 60:.1f} min",
                flush=True,
            )

        if (episode + 1) % CHECKPOINT_EVERY == 0:
            save_weights(q_net, MODEL_PATH)
            np.save(RETURNS_PATH, np.array(rewards_all_episodes))

        if len(rewards_all_episodes) >= 20 and recent_mean >= STOP_AT_MEAN:
            print(f"Episode {episode + 1}: mean20={recent_mean:.1f} >= {STOP_AT_MEAN}, stopping.", flush=True)
            break

    training_minutes = (time.time() - start_time) / 60
    print(f"Training time: {training_minutes:.2f} minutes", flush=True)
    print(f"Environment steps: {total_steps}", flush=True)
    print(f"NN training updates: {len(losses)}", flush=True)

    save_weights(q_net, MODEL_PATH)
    np.save(RETURNS_PATH, np.array(rewards_all_episodes))
    save_learning_curve(rewards_all_episodes, CURVE_PATH)
    print("Saved:", MODEL_PATH, RETURNS_PATH, CURVE_PATH, flush=True)

    # ------------------------------------------------------------------ evaluation
    q_net.eval()
    evaluation_returns = []

    for evaluation_episode in range(EVAL_EPISODES):
        state, info = env.reset(seed=SEED + 1000 + evaluation_episode)
        episode_return = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy(env, q_net, state, EVAL_EPSILON, device)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward

            if terminated or truncated:
                break

        evaluation_returns.append(episode_return)

    evaluation_returns = np.array(evaluation_returns)
    print(f"Evaluation ({EVAL_EPISODES} episodes, epsilon = {EVAL_EPSILON}):", flush=True)
    print(f"Mean return:     {evaluation_returns.mean():.2f}", flush=True)
    print(f"Standard dev.:   {evaluation_returns.std():.2f}", flush=True)
    print(f"Minimum return:  {evaluation_returns.min():.0f}", flush=True)
    print(f"Maximum return:  {evaluation_returns.max():.0f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()

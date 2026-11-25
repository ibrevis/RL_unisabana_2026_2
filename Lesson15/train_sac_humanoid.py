"""Train SAC on Humanoid-v5 with Stable-Baselines3 (training only).

Script version of the training part of Example_04_SAC_Humanoid_SB3.ipynb, meant
for a remote machine with a GPU. It writes, in the current directory:

    sac_humanoid_policy.zip          the trained agent (same name as in the notebook)
    sac_humanoid.monitor.csv         return (r), length (l) and time (t) of every training episode
    checkpoints/sac_humanoid_*.zip   the agent every --checkpoint-freq steps

Usage:
    python train_sac_humanoid.py
    python train_sac_humanoid.py --timesteps 2000000 --device cuda
    nohup python train_sac_humanoid.py > train_sac_humanoid.log 2>&1 &

Requires:
    pip install stable-baselines3 "gymnasium[mujoco]"
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# Training configuration (as in Example_04_SAC_Humanoid_SB3.ipynb)
ENV_ID = "Humanoid-v5"
LEARNING_STARTS = 10000     # random steps that fill the replay buffer before training
BUFFER_SIZE = 1000000      # capacity of the replay buffer
BATCH_SIZE = 256             # transitions per update
GAMMA = 0.99                 # discount factor
LEARNING_RATE = 0.0003       # actor, critics and entropy temperature
TAU = 0.005                  # soft update of the target critics

MODEL_PATH = "sac_humanoid_policy"
MONITOR_PATH = "sac_humanoid"   # Monitor adds ".monitor.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help='"auto" (GPU if available), "cuda" or "cpu"',
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="save the agent every this many environment steps",
    )
    args = parser.parse_args()

    # float32 observations halve the memory of the replay buffer (about 2.8 GB)
    env = Monitor(
        gym.wrappers.DtypeObservation(gym.make(ENV_ID), np.float32),
        filename=MONITOR_PATH,
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    if model.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(model.device)}")

    checkpoints = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="checkpoints",
        name_prefix="sac_humanoid",
    )

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, log_interval=500, callback=checkpoints)
    training_minutes = (time.time() - start_time) / 60

    model.save(MODEL_PATH)
    returns = env.get_episode_rewards()
    env.close()

    print(f"Training time: {training_minutes:.1f} minutes")
    print(f"Environment steps: {model.num_timesteps}")
    print(f"Updates: {model._n_updates}")
    print(f"Mean return, last 20 episodes: {np.mean(returns[-20:]):.1f}")
    print(f"Saved: {MODEL_PATH}.zip and {MONITOR_PATH}.monitor.csv")


if __name__ == "__main__":
    main()

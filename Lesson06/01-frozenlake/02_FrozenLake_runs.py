import gymnasium as gym
import time

env = gym.make("FrozenLake-v1",
               map_name="4x4", #"8x8",
               is_slippery=False,
               reward_schedule=(1, -1, 0),   # (goal, hole, step)
               render_mode="human")

for episode in range(5):
    state, _ = env.reset()
    episode_return = 0.
    for step in range(100):
        action = env.action_space.sample()  # Take a random action
        state, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        time.sleep(0.05)
        if terminated or truncated:
            time.sleep(2)
            break
    print(f"Episode: {episode}, Total Return: {episode_return}")
env.close()
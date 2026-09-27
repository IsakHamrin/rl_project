import pickle
import gymnasium as gym
import numpy as np
import pygame

# 1. Load saved model
model_path = "cart_w_pole/cartpole_q_table.pkl"
with open(model_path, "rb") as f:
    checkpoint = pickle.load(f)

q_values = checkpoint["q_values"]
bins = checkpoint["bins"]


def discretize(obs: np.ndarray) -> tuple[int, ...]:
  # Map continuous state to discrete bins using saved partition thresholds
  return tuple(
      int(np.digitize(feature, bins[i])) for i, feature in enumerate(obs)
  )


def get_best_action(obs: np.ndarray) -> int:
  state = discretize(obs)
  # Default to zeros if encountering an unseen bucket state
  action_values = q_values.get(state, np.zeros(2))
  return int(np.argmax(action_values))


# 2. Run visual evaluation in CartPole
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)
clock = pygame.time.Clock()

num_episodes = 15

for ep in range(num_episodes):
  obs, _ = env.reset()
  done = False
  total_reward = 0

  while not done:
    # Process macOS window events to prevent freezing
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True

    # Pure greedy action selection
    action = get_best_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    done = done or terminated or truncated

    # Cap to 50 FPS (native CartPole physics rate)
    clock.tick(50)

  print(f"Test Episode {ep + 1}: Total Reward = {total_reward}")

env.close()
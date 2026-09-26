# import gymnasium as gym
# import numpy as np
# from collections import defaultdict

# class PoleCartAgent:
#     def __init__(
#         self,
#         env: gym.Env,
#         learning_rate: float = 1e-2,
#         initial_epsilon: float = 1.0,
#         epsilon_decay: float = 2e-5,
#         final_epsilon: float =0.1 ,
#         discount_factor: float = 0.95,
#         n_bins: tuple[int, int, int, int] = (6, 12, 12, 12)
#     ):
#         self.env = env
        
#         # Q-table: maps (state, action) to expected reward
#         # defaultdict automatically creates entries with zeros for new states
#         self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

#         self.lr = learning_rate
#         self.discount_factor = discount_factor  # How much we care about future rewards

#         # Exploration parameters
#         self.epsilon = initial_epsilon
#         self.epsilon_decay = epsilon_decay
#         self.final_epsilon = final_epsilon

#         # Track learning progress
#         self.training_error = []

#         self.bins = [
#                 np.linspace(-2.4, 2.4, n_bins[0] - 1),
#                 np.linspace(-3.0, 3.0, n_bins[1] - 1),
#                 np.linspace(-0.25, 0.25, n_bins[2] - 1),
#                 np.linspace(-3.0, 3.0, n_bins[3] - 1),
#             ]

#     def discretize(self, obs: np.ndarray) -> tuple[int, ...]:
#         # Map continuous float values into discrete bucket indices
#         return tuple(
#             int(np.digitize(feature, self.bins[i]))
#             for i, feature in enumerate(obs)
#         )

#     def get_action(self, obs: np.ndarray) -> int:
#         """Choose an action using epsilon-greedy strategy.

#         Returns:
#             action: 0 (stand) or 1 (hit)
#         """
#         state = self.discretize(obs)
#         # With probability epsilon: explore (random action)
#         if np.random.random() < self.epsilon:
#             return self.env.action_space.sample()

#         # With probability (1-epsilon): exploit (best known action)
#         else:
#             return int(np.argmax(self.q_values[state]))
        
#     def update(
#         self,
#         obs: tuple[int, int, bool],
#         action: int,
#         reward: float,
#         terminated: bool,
#         next_obs: np.ndarray,
#     ):
#         """Update Q-value based on experience.

#         This is the heart of Q-learning: learn from (state, action, reward, next_state)
#         """
#         state = self.discretize(obs)
#         next_state = self.discretize(next_obs)
#                     # What's the best we could do from the next state?
#         # (Zero if episode terminated - no future rewards possible)
#         future_q_value = (not terminated) * np.max(self.q_values[next_state])

#         # What should the Q-value be? (Bellman equation)
#         target = reward + self.discount_factor * future_q_value

#         # How wrong was our current estimate?
#         temporal_difference = target - self.q_values[state][action]

#         # Update our estimate in the direction of the error
#         # Learning rate controls how big steps we take
#         self.q_values[state][action] = (
#             self.q_values[state][action] + self.lr * temporal_difference
#         )

#         # Track learning progress (useful for debugging)
#         self.training_error.append(temporal_difference)

#     def decay_epsilon(self):
#         """Reduce exploration rate after each episode."""
#         self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# # Training hyperparameters
# learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
# n_episodes = 10000        # Number of hands to practice
# start_epsilon = 1.0         # Start with 100% random actions
# epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
# final_epsilon = 0.1         # Always keep some exploration

# # Create environment and agent
# env = gym.make("CartPole-v1")#, render_mode="human")
# env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# agent = PoleCartAgent(
#     env=env,
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

# env.observation_space

# from tqdm import tqdm  # Progress bar

# for episode in tqdm(range(n_episodes)):
#     # Start a new hand
#     obs, info = env.reset()
#     done = False

#     # Play one complete hand
#     while not done:
#         # Agent chooses action (initially random, gradually more intelligent)
#         action = agent.get_action(obs)

#         # Take action and observe result
#         next_obs, reward, terminated, truncated, info = env.step(action)

#         # Learn from this experience
#         agent.update(obs, action, reward, terminated, next_obs)

#         # Move to next state
#         done = terminated or truncated
#         obs = next_obs

#     # Reduce exploration rate (agent becomes less random over time)
#     agent.decay_epsilon()


# eval_env = gym.make("CartPole-v1", render_mode="human")
# eval_episodes = 50
# eval_rewards = []

# # Turn off exploration for evaluation
# saved_epsilon = agent.epsilon
# agent.epsilon = 0.0

# for ep in range(eval_episodes):
#     obs, _ = eval_env.reset()
#     done = False
#     ep_reward = 0

#     while not done:
#         action = agent.get_action(obs)
#         obs, reward, terminated, truncated, _ = eval_env.step(action)
#         ep_reward += reward
#         done = terminated or truncated

#     eval_rewards.append(ep_reward)

# eval_env.close()
# agent.epsilon = saved_epsilon

# print(f"Evaluation over {eval_episodes} episodes:")
# print(f"Mean Return : {np.mean(eval_rewards):.1f} +/- {np.std(eval_rewards):.1f}")
# print(f"Max Return  : {np.max(eval_rewards):.1f}")
# print(f"Min Return  : {np.min(eval_rewards):.1f}")

import pickle
import gymnasium as gym
import numpy as np
import pygame

# 1. Load saved model
model_path = "rl_project/cart_w_pole/cartpole_q_table.pkl"
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
env = gym.make("CartPole-v1", render_mode="human")
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
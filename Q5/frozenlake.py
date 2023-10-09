import gym
import numpy as np

# Create a Frozen Lake environment with a random 8x8 map
env = gym.make("FrozenLake8x8-v1")

# Initialize Q-table with zeros
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 10000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-value using Q-learning equation
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Evaluate the trained agent
num_evaluation_episodes = 100
total_rewards = []
for _ in range(num_evaluation_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    total_rewards.append(total_reward)
average_reward = np.mean(total_rewards)
print(f"Average Reward over {num_evaluation_episodes} evaluation episodes: {average_reward}")



class CustomFrozenLake(gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
    def __init__(self, desc=None, map_name="8x8", is_slippery=True):
        super().__init__(desc=desc, map_name=map_name, is_slippery=is_slippery)

    def step(self, a):
        if self.np_random.rand() < 0.02:
            # 2% chance of randomly dying of cold
            return 0, self.nS - 1, True, {}
        else:
            return super().step(a)


# Create a custom Frozen Lake environment with random death chance
env_with_death_chance = CustomFrozenLake()

import math
from typing import Optional, Union
import numpy as np
import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
import time
import gym.wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
from controller import pid_controller
from cartpole import CartPoleEnv


def trainAgent():
    # Step 1: Create the CartPole Environment
    env = CartPoleEnv()

    # Step 3: Initialize and Train the RL Agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(
        total_timesteps=100000, progress_bar=True
    )  # Adjust the number of timesteps as needed

    # output to see how well we are doing
    mean_reward = evaluateAgent(env, model)

    # Step 4: Save the Trained Agent
    model.save("ppo_cartpole")
    env.close()


def evaluateAgent(env, model):
    epsilon = 0.2
    total_reward = 0
    iterations = 10
    for _ in range(iterations):
        obs = env.reset()
        done = False
        while not done:
            # epsilon greedy action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    average_reward = total_reward / iterations
    return average_reward


def loadTrainedAgent(episodes=10, test=False):
    import gym
    from stable_baselines3 import PPO

    # Create the CartPole environment
    env = CartPoleEnv(render_mode="human")

    # Load the saved model
    model = PPO.load("ppo_cartpole")

    # Define the number of episodes to run
    for episode in range(episodes):
        state = env.reset()
        if len(state) == 2:
            state = state[0]
        total_reward = 0
        steps = 0
        while True:
            # Use the trained model to choose an action
            action, _ = model.predict(state)
            # Step forward in the environment
            state, reward, done, truncated, _ = env.step(action)
            steps += 1
            # Accumulate total reward
            total_reward += reward
            if done:
                break
            # if test and steps > 200:
            #     break
    # Close the environment when done
    env.close()
    return steps


if __name__ == "__main__":
    # trainAgent()
    trainedSteps = loadTrainedAgent(episodes=20, test=True)
    # print(f"Trained agent took {trainedSteps} steps")
    # kp, ki, kd = developControllerRatios()
    # avgTime, avgSteps = pid_controller(17, 0, 19, episodes=5, human=True)
    # print(f"PID controller took on average {avgSteps} steps")

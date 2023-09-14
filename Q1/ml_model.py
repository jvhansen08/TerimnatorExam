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
import optuna


def evaluateAgent(env, model):
    epsilon = 0.2
    total_reward = 0
    iterations = 10
    for _ in range(iterations):
        obs = env.reset()
        if len(obs) == 2:
            obs = obs[0]
        done = False
        while not done:
            # epsilon greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

    average_reward = total_reward / iterations
    return average_reward


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    n_steps = trial.suggest_int("n_steps", 32, 512)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    env = CartPoleEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    model.learn(total_timesteps=10000, progress_bar=True)
    mean_reward = evaluateAgent(env, model)
    env.close()
    return mean_reward


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


def getBestTrainingParams():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_learning_rate = best_params["learning_rate"]
    best_n_steps = best_params["n_steps"]
    best_batch_size = best_params["batch_size"]
    print(f"Best params: {best_params}")


def trainAgent(learning_rate=0.0008607377834906738, n_steps=46, batch_size=337):
    env = CartPoleEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("ppo_cartpole")
    env.close()


if __name__ == "__main__":
    # getBestTrainingParams()
    trainAgent()
    trainedSteps = loadTrainedAgent(episodes=20, test=True)
    print(f"Trained agent took {trainedSteps} steps")
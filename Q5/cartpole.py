import gym
import numpy as np
import tensorflow as tf
import gym.wrappers
from rightForcefulCartpole import RightCartPoleEnv
from stable_baselines3 import PPO


def trainAgent(env):
    print("Training model")
    model = PPO("MlpPolicy", env, verbose=1, seed=80)
    model = model.learn(total_timesteps=50000, progress_bar=True)
    model.save("ppo_right_cartpole")
    return model


def evaluateModel(env, model):
    print("Evaluating model")
    total_reward = 0
    num_eval_episodes = 10
    for _ in range(num_eval_episodes):
        localReward = 0
        state = env.reset(seed=80)
        if len(state) == 2:
            state = state[0]
        done = False
        while not done and localReward < 500:
            action, _ = model.predict(state)
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            localReward += reward
            state = next_state

    average_reward = total_reward / num_eval_episodes
    print(
        f"Average Reward over {num_eval_episodes} evaluation episodes: {average_reward}"
    )
    env.close()


def displayModel(env=RightCartPoleEnv(), model=PPO.load("ppo_right_cartpole")):
    # Display the trained model
    print("Displaying model")
    env.render_mode = "human"
    state = env.reset(seed=80)
    if len(state) == 2:
        state = state[0]
        done = False
        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, info, _ = env.step(action)
            state = next_state
    env.close()


if __name__ == "__main__":
    # Create the CartPole-v1 environment
    env = RightCartPoleEnv()
    # model = trainAgent(env)
    model = PPO.load("ppo_right_cartpole")
    evaluateModel(env, model)
    displayModel(env, model)

import numpy as np
from stable_baselines3 import PPO
from cartpole import CartPoleEnv
import optuna
from cartpole import SEED


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


def getBestTrainingParams():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    best_learning_rate = best_params["learning_rate"]
    best_n_steps = best_params["n_steps"]
    best_batch_size = best_params["batch_size"]
    print(f"Best params: {best_params}")
    return best_learning_rate, best_n_steps, best_batch_size


def trainAgent():
    # learning_rate=0.0005754,
    # gamma=0.912318,
    # n_steps=478,
    # batch_size=478 * 4,
    env = CartPoleEnv()
    model = PPO("MlpPolicy", env, verbose=1, seed=85)
    model = model.learn(total_timesteps=100000, progress_bar=True)
    model.save("ppo_cartpole_85")
    env.close()


def evaluateTrainedAgent(episodes=10, maxSteps=None):
    """Evaluate the trained agent and return the pass rate"""
    # Create the CartPole environment
    # Load the saved model
    env = CartPoleEnv()
    model = PPO.load("ppo_cartpole_85")
    stepsCounter = []
    # Define the number of episodes to run
    for episode in range(episodes):
        state = env.reset(seed=SEED)
        if len(state) == 2:
            state = state[0]
        steps = 0
        while True:
            # Use the trained model to choose an action
            action, _ = model.predict(state)
            # Step forward in the environment
            state, rewards, done, info, _ = env.step(action)
            steps += 1
            # Accumulate total reward
            if done:
                break
            if maxSteps and steps > maxSteps:
                break
        stepsCounter.append(steps)
    # Close the environment when done
    env.close()
    failures = countFailures(stepsCounter, maxSteps)
    passRate = round(
        1 - (failures / episodes), 3
    )  # Note this also takes into account where the bot goes off the frame. At about 500 steps if still up it is stable, but it will go off the screen to stay balanced
    print(
        f"Worst: {np.min(stepsCounter)} Best: {np.max(stepsCounter)} Average:{np.mean(stepsCounter)}"
    )
    # print(f"Failures {failures} Pass Rate: {passRate}")
    return passRate


def displayAgent():
    # Create the CartPole environment
    # Load the saved model
    env = CartPoleEnv(render_mode="human")
    model = PPO.load("ppo_cartpole_85")
    stepsCounter = []
    state = env.reset(seed=SEED)
    if len(state) == 2:
        state = state[0]
    steps = 0
    while True:
        # Use the trained model to choose an action
        action, _ = model.predict(state)
        # Step forward in the environment
        state, reward, done, info, _ = env.step(action)
        steps += 1
        # Accumulate total reward
        if done:
            break
    stepsCounter.append(steps)
    # Close the environment when done
    env.close()
    return stepsCounter


def countFailures(steps, maxSteps):
    counter = 0
    for i in steps:
        if i < maxSteps:
            counter += 1
    return counter


if __name__ == "__main__":
    maxSteps = 1000
    episodes = 50
    # trainAgent()
    # evaluateTrainedAgent(episodes=episodes, maxSteps=maxSteps)
    displayAgent()

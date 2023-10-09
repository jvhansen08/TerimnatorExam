import gym
import numpy as np
import tensorflow as tf
import gym.wrappers
from gym.envs.classic_control import cartpole

# Create the CartPole-v1 environment
env = gym.make("CartPole-v1")

# Define neural network model for Q-learning
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(24, input_shape=(4,), activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(
            2, activation="linear"
        ),  # 2 actions: push left or push right
    ]
)
# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.mean_squared_error


def trainModel(save=True):
    # Define training parameters
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration

    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        if len(state) == 2:
            state = state[0]
        state = np.reshape(state, [1, 4])

        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                Q_values = model.predict(state)
                action = np.argmax(Q_values)  # Exploitation
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # Calculate target Q-value
            target = reward + gamma * np.max(model.predict(next_state))

            # Calculate loss and perform gradient descent
            with tf.GradientTape() as tape:
                Q_values = model(state)
                loss = loss_fn(Q_values, target)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_reward += reward
            state = next_state

        # Decay exploration rate
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    if save:
        # Save the trained model
        model.save("cartpole_model.h5")


# Evaluate the trained model
def evaluateModel():
    total_reward = 0
    num_eval_episodes = 10
    for _ in range(num_eval_episodes):
        state = env.reset()
        if len(state) == 2:
            state = state[0]
        state = np.reshape(state, [1, 4])
        done = False
        while not done:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            total_reward += reward
            state = next_state

    average_reward = total_reward / num_eval_episodes
    print(
        f"Average Reward over {num_eval_episodes} evaluation episodes: {average_reward}"
    )
    env.close()


def displayModel():
    # Display the trained model
    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()
    state = np.reshape(state, [1, 4])
    done = False
    while not done:
        action = np.argmax(model.predict(state))
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
    env.close()

class CustomCartpole(gym.envs.toy_text.cartpole.CartPoleEnv):
    def __init__(self, desc=None):
        super().__init__(desc=desc, )

        


if __name__ == "__main__":
    print("Training model")
    trainModel()
    print("Evaluating model")
    evaluateModel()
    print("Done!")
    print("Displaying model")
    displayModel()

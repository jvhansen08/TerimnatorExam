from cartpole import CartPoleEnv
import numpy as np
from datetime import datetime


def pid_controller(Kp, Ki, Kd, episodes, human=False, demo=False):
    # Create the CartPole environment
    if human:
        env = CartPoleEnv(render_mode="human")
    else:
        env = CartPoleEnv()
    # Initialize PID controller variables
    integral = 0
    prev_error = 0
    # Simulation parameters
    rewards = []
    steps = []
    for episode in range(episodes):
        if demo:
            print("Episode: ", episode)
        state = env.reset()
        reward = 0
        done = False
        stepCounter = 0
        while not done and stepCounter < 5000:
            # Extract state information
            if len(state) == 2:
                array = state[0]
                pole_angle = array[2]
            else:
                pole_angle = state[2]
            # Calculate error (difference from the upright position)
            error = pole_angle
            # Update integral term
            integral += error
            # Calculate control action (PID controller)
            control_action = Kp * error + Ki * integral + Kd * (error - prev_error)
            # Apply the control action (push cart left or right)
            if control_action > 0:
                action = 1  # Push cart to the right
            else:
                action = 0  # Push cart to the left
            # Step forward in the environment
            state, newR, done, truncated, _ = env.step(action)
            stepCounter += 1
            # Update previous error
            prev_error = error
            # Accumulate total reward
            reward += newR
        rewards.append(reward)
        steps.append(stepCounter)
    env.close()
    return np.mean(rewards), np.mean(steps)


def developControllerRatios():
    bestTime = 0
    bestTimeValues = [0] * 3
    bestSteps = 0
    bestStepsValues = [0] * 3
    averageTimes = []
    iterations = 20
    episodes = 10
    start = datetime.now()
    for kp in range(iterations):  # Proportional gain
        for ki in range(iterations):  # Integral gain
            for kd in range(iterations):  # Derivative gain
                avgTime, avgSteps = pid_controller(kp, ki, kd, episodes=episodes)
                averageTimes.append(avgTime)
                if avgTime > bestTime:
                    bestTime = avgTime
                    bestTimeValues = [kp, ki, kd]
                    print(f"New best time: {bestTime} with values {bestTimeValues}")
                if avgSteps > bestSteps:
                    bestSteps = avgSteps
                    bestStepsValues = [kp, ki, kd]
                    print(f"New best steps: {bestSteps} with values {bestStepsValues}")
    finish = datetime.now()
    print(
        f"Time taken: {finish - start} with {episodes} episodes and {iterations} iterations"
    )
    print(bestTime, bestTimeValues)

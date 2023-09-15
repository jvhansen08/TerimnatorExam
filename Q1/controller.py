from cartpole import CartPoleEnv
import numpy as np
from datetime import datetime


def evaluatePIDController(Kp, Ki, Kd, episodes, human=False, maxSteps=None):
    # Create the CartPole environment
    if human:
        env = CartPoleEnv(render_mode="human")
    else:
        env = CartPoleEnv()
    # Initialize PID controller variables
    integral = 0
    prev_error = 0
    # Simulation parameters
    steps = np.array([])
    for episode in range(episodes):
        state = env.reset()
        reward = 0
        done = False
        stepCounter = 0
        while not done:
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
            if maxSteps and stepCounter > maxSteps:
                done = True
        steps = np.append(steps, stepCounter)
    env.close()
    return np.median(steps)


def developControllerRatios():
    bestTime = 0
    bestTimeValues = [0] * 3
    bestSteps = 0
    bestStepsValues = [0] * 3
    iterations = 50
    episodes = 20
    start = datetime.now()
    ki = 0
    for kp in range(iterations):  # Proportional gain
        if kp % 10 == 0:
            print(f"kp: {kp}")
        for kd in range(iterations):  # Derivative gain
            avgSteps = evaluatePIDController(kp, ki, kd, episodes=episodes)
            if avgSteps > bestSteps:
                bestSteps = avgSteps
                bestStepsValues = [kp, ki, kd]
                print(f"New best steps: {bestSteps} with values {bestStepsValues}")
    finish = datetime.now()
    print(
        f"Time taken: {finish - start} with {episodes} episodes and {iterations} iterations"
    )
    print(bestTime, bestTimeValues)
    return bestTimeValues


if __name__ == "__main__":
    # values = developControllerRatios()
    values = [17, 0, 33]  # best values out of 100 kps x 100 kds
    controllerStepsAverage = evaluatePIDController(
        values[0], values[1], values[2], human=True, episodes=10, maxSteps=300
    )
    print(f"Controller took on average {controllerStepsAverage} steps")

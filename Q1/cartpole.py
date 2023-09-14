"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
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


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 3.0
        self.masspole = 0.0
        self.massball = 0.5  # weight on top of the pole
        self.total_mass = self.masspole + self.masscart + self.massball
        self.length = 0.4  # actually half the pole's length
        self.polemass_length = (
            self.masspole + self.massball
        ) * self.length  # used for equations later
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode

        degreesTilt = 80  # defaults to 12
        self.theta_threshold_radians = (
            degreesTilt * math.pi / 180
        )  # how far of a lean to stop at
        self.x_threshold = (
            3  # how much space the robot can slide left and right. defaults to 2.4
        )

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        self.maxRecoverable = 0

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # These are the equations from the actual paper, (23-24)
        temp = (
            -force - self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta + costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = (
            force
            + self.polemass_length
            * (theta_dot**2 * sintheta - theta_dot * costheta)
            / self.total_mass
        )

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        degrees = theta * 180 / math.pi
        if not terminated and degrees > self.maxRecoverable:
            self.maxRecoverable = degrees
            # print(f"Max recoverable = {degrees}")

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def trainAgent():
    # Step 1: Create the CartPole Environment
    env = CartPoleEnv()

    # Step 3: Initialize and Train the RL Agent
    model = PPO("MlpPolicy", env, verbose=1, batch_size=100)
    model.learn(
        total_timesteps=100000, log_interval=5000
    )  # Adjust the number of timesteps as needed

    # Step 4: Save the Trained Agent
    model.save("ppo_cartpole")
    env.close()


def loadTrainedAgent(episodes=10):
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
        while True and steps < 5000:
            # Use the trained model to choose an action
            action, _ = model.predict(state)
            # Step forward in the environment
            state, reward, done, truncated, _ = env.step(action)
            steps += 1
            # Accumulate total reward
            total_reward += reward
            if done:
                break
    # Close the environment when done
    env.close()
    return steps


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
    iterations = 10
    episodes = 50
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


def noTraining():
    env = CartPoleEnv(render_mode="human")
    env.reset()
    for _ in range(1000):
        state, newR, done, truncated, _ = env.step(env.action_space.sample())
        if done:
            break
        env.render()
        time.sleep(0.1)
    env.close()


def demo():
    # print("\n---Demoing cart pole problem---\n")
    # noTraining()
    # print("\n---Demoing trained agent---\n")
    # loadTrainedAgent(episodes=3)
    print("\n---Demoing PID controller---\n")
    pid_controller(1, 0, 4, episodes=10, human=True, demo=True)


if __name__ == "__main__":
    demo()
    # trainAgent()
    # trainedSteps = loadTrainedAgent()
    # print(f"Trained agent took {trainedSteps} steps")
    # kp, ki, kd = developControllerRatios()
    # avgTime, avgSteps = pid_controller(1, 0, 4, episodes=5, human=True)
    # print(f"PID controller took on average {avgSteps} steps")

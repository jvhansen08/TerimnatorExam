import numpy as np
import matplotlib.pyplot as plt


def ackermannCircle():
    # TODO: Figure out how to start from the center and plot the truck path's over the circle's
    # Conversion factor from feet to meters
    feet_to_meters = 0.3048
    # Constants
    robot_length = 35.0 / feet_to_meters  # in meters
    robot_width = 10.0 / feet_to_meters  # in meters
    # Constants
    circle_radius = 18.0  # in meters
    velocity = 8.0  # in m/s
    frequency = 2.0  # Hz

    # Calculate the circumference of the circle
    circle_circumference = 2 * np.pi * circle_radius

    # Calculate the time it takes to complete one circle
    circle_time = circle_circumference / velocity

    # Calculate the number of points to simulate at 2Hz
    num_points = int(circle_time * frequency)

    # Initialize arrays to store x and y coordinates
    x_coordinates = []
    y_coordinates = []
    angular_velocities = []

    # Calculate the angle increment for each time step
    angle_increment = 2 * np.pi / num_points

    # Simulate the motion starting at the edge of the circle
    for i in range(num_points):
        angle = i * angle_increment
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)
        x_coordinates.append(x)
        y_coordinates.append(y)
    plot(x_coordinates, y_coordinates, angular_velocities)

def skidSteer():
    # TODO: Implement the same circle mechanism, but using skid steer insted of the ackerman model
    pass

def positionalError():
    pass
    #TODO: Use Euler method to calculate the positional error of the robot
    # TODO: Graph the rrors for 3 time steps 1, 0.1, and 0.01

def plot(x_coordinates, y_coordinates, angular_velocities):

    # Plot the circle
    circle = plt.Circle((0, 0), 18, color="black", fill=False)


    # Plot the path
    plt.figure(figsize=(10, 6))
    plt.plot(x_coordinates, y_coordinates, label="Path", color="blue")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Robot Path along the Edge of a Circle")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the angular velocities
    plt.figure(figsize=(10, 4))
    plt.plot(angular_velocities)
    plt.xlabel("Time Steps")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("Angular Velocity over Time")
    plt.grid(True)
    plt.show()


def main():
    ackermannCircle()


if __name__ == "__main__":
    main()

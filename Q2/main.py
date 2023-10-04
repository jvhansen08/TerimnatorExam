import matplotlib.pyplot as plt
import numpy as np
import time

FEET_TO_METERS = 0.3048
VEHICLE_LENGTH = 35 / FEET_TO_METERS  # length of the vehicle
VEHICLE_WIDTH = 10 / FEET_TO_METERS  # width of the vehicle
VEHICLE_VELOCITY = 8  # m/s
UPDATE_SPEED = 0.5  # 2 hz


def ackermanCircle():
    x_coordinates = [0]
    y_coordinates = [0]
    theta_history = [0]
    # Move robot to get to the edge of the circle
    getToEdge(x_coordinates, y_coordinates, theta_history)
    moveCircle(x_coordinates, y_coordinates, theta_history)
    plt.plot(x_coordinates, y_coordinates)
    # plt.savefig("ackerman.png")
    plt.show()


def getToEdge(x_coordinates, y_coordinates, theta_history):
    innerRadius = 9
    # alpha is the inverse tangent of the ratio of the length of the vehicle to the radius of the circle
    alpha = np.arctan(VEHICLE_LENGTH / innerRadius)  # equation 1.1
    seconds = 2 * np.pi * innerRadius / VEHICLE_VELOCITY  # 2 pi r / v
    moveRobot(x_coordinates, y_coordinates, theta_history, seconds / 2, alpha)


def moveCircle(x_coordinates, y_coordinates, theta_history, radius=18):
    seconds = 2 * np.pi * radius / VEHICLE_VELOCITY  # 2 pi r / v
    alpha = np.arctan(VEHICLE_LENGTH / radius)  # equation 1.1
    moveRobot(x_coordinates, y_coordinates, theta_history, seconds, alpha)


def moveRobot(x_coordinates, y_coordinates, theta_history, number_seconds, alpha):
    # Equations are from Mobile-Robots
    for _ in range(int(number_seconds / UPDATE_SPEED)):
        x_velocity = -VEHICLE_VELOCITY * np.sin(theta_history[-1])  # x dot
        y_velocity = VEHICLE_VELOCITY * np.cos(theta_history[-1])  # y dot
        omega = VEHICLE_VELOCITY / VEHICLE_LENGTH * np.tan(alpha)  # angular velocity
        x_next = x_coordinates[-1] + x_velocity * UPDATE_SPEED
        y_next = y_coordinates[-1] + y_velocity * UPDATE_SPEED
        theta_next = theta_history[-1] + omega * UPDATE_SPEED
        x_coordinates.append(x_next)
        y_coordinates.append(y_next)
        theta_history.append(theta_next)


def skidSteer():
    # Skid steer setup will just be move to edge, rotate 90 degrees, then move in a circle
    x_coordinates = [0]
    y_coordinates = [0]
    theta_history = [0]
    # Move robot to get to the edge of the circle
    skidSteerGetToEdge(x_coordinates, y_coordinates, theta_history)
    # Now rotate 90 degrees
    theta_history[-1] += np.pi / 2
    x_coordinates.append(x_coordinates[-1])
    y_coordinates.append(y_coordinates[-1])
    # Complete the circle
    skidSteerMoveCircle(x_coordinates, y_coordinates, theta_history)
    plt.plot(x_coordinates, y_coordinates)
    # plt.savefig("skidSteer.png")
    plt.show()


def skidSteerGetToEdge(x_coordinates, y_coordinates, theta_history):
    moveRobot(x_coordinates, y_coordinates, theta_history, 2, 0)


def skidSteerMoveCircle(x_coordinates, y_coordinates, theta_history, radius=18):
    # Equations are from Mobile-Robots 1.11a, 1.11b, and 1.11c
    # Apply equal force to both wheels
    seconds = 2 * np.pi * radius / VEHICLE_VELOCITY  # 2 pi r / v
    alpha = np.arctan(VEHICLE_LENGTH / radius)
    phi_dot = VEHICLE_VELOCITY / VEHICLE_LENGTH * np.tan(alpha)  # angular velocity
    leftV = phi_dot * (radius - VEHICLE_WIDTH / 2)  # 1.7a
    rightV = phi_dot * (radius + VEHICLE_WIDTH / 2)  # 1.7b
    for _ in range(int(seconds / UPDATE_SPEED)):
        x_velocity = -(rightV + leftV) * np.sin(theta_history[-1]) / 2  # x dot 1.11a
        y_velocity = (rightV + leftV) * np.cos(theta_history[-1]) / 2  # y dot 1.11b
        omega = (rightV - leftV) / VEHICLE_WIDTH  # angular velocity
        x_next = x_coordinates[-1] + x_velocity * UPDATE_SPEED
        y_next = y_coordinates[-1] + y_velocity * UPDATE_SPEED
        theta_next = theta_history[-1] + omega * UPDATE_SPEED
        x_coordinates.append(x_next)
        y_coordinates.append(y_next)
        theta_history.append(theta_next)


def positionalError():
    # Initialize variables:
    radius = 9  # meters
    circleSeconds = 2 * np.pi * radius / VEHICLE_VELOCITY  # 2 pi r / v
    # Get movement data
    x_coordinates = [0]
    y_coordinates = [0]
    theta_history = [0]
    skidSteerMoveCircle(x_coordinates, y_coordinates, theta_history, radius)
    # Now we compare the robot movement to our ground truth movement at various time intervals
    differenceCollections = []
    intervals = [1, 0.5, 0.2, 0.15, 0.1, 0.01]
    for interval in intervals:
        start = time.time()
        differences = errorAtIncrement(
            interval, x_coordinates, y_coordinates, radius, circleSeconds
        )
        differenceCollections.append(differences)
        end = time.time()
        print(f"Time taken: {round(end - start, 2)} with interval {interval}")
        print(f"Total Error: {sum(differences)} Max error: {max(differences)}")

    # Plot the differences
    for differences, interval in zip(differenceCollections, intervals):
        plt.plot(
            [i * interval for i in range(int(circleSeconds / interval))],
            differences,
            label=f"Interval of {interval} seconds",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Max Error (meters)")
        plt.ylim(0, max(differences) + np.median(differences) / 4)
        plt.legend()
    # plt.show()

    # plt.savefig(f"totalErrors.png")


def errorAtIncrement(
    incrementSize, x_coordinates, y_coordinates, radius, circleSeconds
):
    # We know radius in len(x_coordinates) increments. Convert this to its appropriate interval
    experimentalTimestamp = int(
        incrementSize / UPDATE_SPEED
    )  # measured at update speed for every second. We have 2 measurements per second
    # Now at each time interval compare computed value to the analytical value
    errors = []
    for time in range(int(circleSeconds / incrementSize)):
        computedX, computedY = circle_position(
            radius, VEHICLE_VELOCITY, time * incrementSize, x0=-radius, y0=0
        )
        experimentalX, experimentalY = (
            x_coordinates[time * experimentalTimestamp],
            y_coordinates[time * experimentalTimestamp],
        )
        error = np.sqrt(
            (experimentalX - computedX) ** 2 + (experimentalY - computedY) ** 2
        )
        errors.append(error)
    return errors


def circle_position(radius, velocity, t, x0=0, y0=0):
    """
    Calculate the exact x and y coordinates of a point moving in a circle.

    Args:
    - radius (float): The radius of the circle.
    - velocity (float): The constant velocity of the point.
    - t (float): The time at which to calculate the position.

    Returns:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    """

    # At time zero this should be returning 0,0
    angular_velocity = velocity / radius  # Angular velocity in radians per second
    angle = angular_velocity * t  # Angle in radians at time 't'
    x = radius * np.cos(angle) + x0
    y = radius * np.sin(angle) + y0
    return x, y


def main():
    """Notes from Lectures:
    - September 8 2023
        - The length of the robot matters A LOT
        - Whe we start, what is in front of me is y ground
        - When we turn, consider where am I in my own udnerstand and where am I in relation to the map
        Timestamp at 35:00 - 40:00
        - Geometry gives us L/R = tan(alpha), therefore the R = L/tan(alpha)
        - The real world for phi dot, change in global theta, is phi dot = v_rearWheel/R
        - Equations 1.4 a b c are all the equations we need for question 2
        - Equations 1.11a, 1.11b, and 1.11c will help for the skid steer question (part b)
    - September 11 2023
        - For part 3 we know the circle and the velocity of the robot, meaning at every time step we know what the ground truth is
        - Now we simply go along the circle and calculate the error at each time step, and graph it
    """
    # ackermanCircle()
    # skidSteer()
    positionalError()


if __name__ == "__main__":
    main()

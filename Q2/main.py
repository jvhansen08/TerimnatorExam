import numpy as np
import matplotlib.pyplot as plt


# Discussed in Lecture on September 8 2023, September 11 2023

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
import math
import random
import numpy as np
import matplotlib.pyplot as plt
FEET_TO_METERS = 0.3048

VEHICLE_LENGTH = 35 / FEET_TO_METERS # length of the vehicle
VEHICLE_WIDTH = 10 / FEET_TO_METERS # width of the vehicle
VEHICLE_VELOCITY = 8 # m/s
UPDATE_SPEED = 0.5 # 2 hz
def ackermanCircle():
    ## Questions:
    # Is alpha up to me to decide or can we try different values?
    # How do I keep this within the radius of the circle?
    # Will the skid steer version of this essentially be the same except for the equation modifications? Yes, and to get to the edge we can move forwd.
    # What is the forward Euler method? 
    x_coordinates = [0]
    y_coordinates = [0]
    theta_history = [0]
    # Move robot to get to the edge of the circle
    getToEdge(x_coordinates, y_coordinates, theta_history)
    moveCircle(x_coordinates, y_coordinates, theta_history)
    # TODO: Create a function to up to the top. Calculate radius of inner circle to get to the edge, then move over there before starting
    plt.plot(x_coordinates, y_coordinates)
    plt.show()

def getToEdge(x_coordinates, y_coordinates, theta_history):
    innerRadius = 9
    # alpha is the inverse tangent of the ratio of the length of the vehicle to the radius of the circle
    alpha = np.arctan(VEHICLE_LENGTH/innerRadius) # equation 1.1
    seconds = 2 * np.pi * innerRadius / VEHICLE_VELOCITY # 2 pi r / v
    moveRobot(x_coordinates, y_coordinates, theta_history, seconds/2, alpha)

def moveCircle(x_coordinates, y_coordinates, theta_history):
    radius = 18
    seconds = 2 * np.pi * radius / VEHICLE_VELOCITY # 2 pi r / v
    alpha = np.arctan(VEHICLE_LENGTH/radius) # equation 1.1
    moveRobot(x_coordinates, y_coordinates, theta_history, seconds, alpha)
     
def moveRobot(x_coordinates, y_coordinates, theta_history, number_seconds, alpha):
        for _ in range(int(number_seconds/UPDATE_SPEED)):
            x_velocity = -VEHICLE_VELOCITY*np.sin(theta_history[-1]) # x dot
            y_velocity = VEHICLE_VELOCITY*np.cos(theta_history[-1]) # y dot
            omega = VEHICLE_VELOCITY/VEHICLE_LENGTH*np.tan(alpha) # angular velocity
            x_next = x_coordinates[-1] + x_velocity*UPDATE_SPEED
            y_next = y_coordinates[-1] + y_velocity*UPDATE_SPEED
            theta_next = theta_history[-1] + omega*UPDATE_SPEED
            x_coordinates.append(x_next)
            y_coordinates.append(y_next)
            theta_history.append(theta_next)
        

def skidSteer():
    # Skid steer setup will just be straight line, get to edge, then rotate 90 degrees
    # TODO: Implement the same circle mechanism, but using skid steer insted of the ackerman model
    pass

def positionalError():
    pass
    circleRadius = 9 # meters
    # We know the radius of circle and velocity of robot, meaning at every time step we know what the ground truth is:
    # Now we simply go along the circle and calculate the error at each time step, and graph it
    
    #TODO: Use Euler method to calculate the positional error of the robot
    # TODO: Graph the rrors for 3 time steps 1, 0.1, and 0.01


def main():
    ackermanCircle()


if __name__ == "__main__":
    main()

from Planner import Planner
from Dijkstra import CustomDijkstraPlanner
from AStar import AStarPlanner
import matplotlib.pyplot as plt


def createGrid(show_animation):
    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
    return ox, oy, sx, sy, gx, gy


def main():
    resolution = 1  # [m]
    robot_radius = 1  # [m]
    show_animation = False
    ox, oy, sx, sy, gx, gy = createGrid(True)
    # p = AStarPlanner(ox, oy, resolution, robot_radius)
    # rx, ry = p.plan(sx, sy, gx, gy, show_animation)
    # p.graph(rx, ry)
    for algorithm in [CustomDijkstraPlanner, AStarPlanner]:
        print(algorithm.__name__)
        planner = algorithm(ox, oy, resolution, robot_radius)  # Initialize the planner
        rx, ry = planner.plan(sx, sy, gx, gy, show_animation)
        planner.graph(rx, ry)


if __name__ == "__main__":
    main()

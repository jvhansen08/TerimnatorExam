from Planner import Planner
from Dijkstra import CustomDijkstraPlanner
from AStar import CustomAStarPlanner
from BiDirectional import CustomBiDirectional
from BFSPlanner import CustomBFSPlanner
from RRTStar import CustomRRTStar
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


def runRRtStar(obsList, start, goal, robot_radius, show_animation):
    # RRT Star has different setup
    planner = CustomRRTStar(
        start=start,
        goal=goal,
        obstacle_list=obsList,
        robot_radius=robot_radius,
        rand_area=[-10, 60],
    )
    path = planner.plan(show_animation)
    planner.graph(path)
    print()


def runPlanners(ox, oy, sx, sy, gx, gy, robot_radius, show_animation, resolution):
    for algorithm in [
        CustomDijkstraPlanner,
        CustomAStarPlanner,
        CustomBFSPlanner,
        CustomBiDirectional,
    ]:
        print(algorithm.__name__)
        planner = algorithm(ox, oy, resolution, robot_radius)  # Initialize the planner
        rx, ry = planner.plan(sx, sy, gx, gy, show_animation)
        if show_animation:
            planner.graph(rx, ry)


def main():
    resolution = 1  # [m]
    robot_radius = 1  # [m]
    show_animation = False
    ox, oy, sx, sy, gx, gy = createGrid(True)
    obsList = [[ox[i], oy[i], 1] for i in range(len(ox))]
    # runPlanners(ox, oy, sx, sy, gx, gy, robot_radius, show_animation, resolution)
    runRRtStar(obsList, [sx, sy], [gx, gy], robot_radius, show_animation)


if __name__ == "__main__":
    main()

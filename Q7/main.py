from Dijkstra import CustomDijkstraPlanner
from AStar import CustomAStarPlanner
from BiDirectional import CustomBiDirectional
from BFSPlanner import CustomBFSPlanner
from RRTStar import CustomRRTStar
from AStar_Vanilla import VanillaAStar
import matplotlib.pyplot as plt
import math
import time


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
        plotBoard(ox, oy, sx, sy, gx, gy)
    return ox, oy, sx, sy, gx, gy


def plotBoard(ox, oy, sx, sy, gx, gy):
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")


def runRRtStar(obsList, start, goal, robot_radius, show_animation):
    # RRT Star has different setup
    totalCost = 0
    totalTime = 0
    iterations = 1
    for i in range(iterations):
        planner = CustomRRTStar(
            start=start,
            goal=goal,
            obstacle_list=obsList,
            robot_radius=robot_radius,
            rand_area=[-10, 60],
        )
        plt.clf()
        startTime = time.time()
        path = planner.plan(show_animation)
        finishTime = time.time()
        cost = analyzeCost(path[0], path[1])
        totalCost += cost
        totalTime += finishTime - startTime
        planner.graph(path)
    print(
        f"| RRT Star | {round(totalTime/iterations, 2)} | {round(totalCost/iterations,2)} |"
    )


def runPlanners(
    ox, oy, sx, sy, gx, gy, robot_radius, show_animation, resolution, obsList
):
    for algorithm in [
        # CustomDijkstraPlanner,
        # CustomAStarPlanner,
        # CustomBFSPlanner,
        # CustomBiDirectional,
        VanillaAStar
    ]:
        # Reset the plot
        totalCost = 0
        totalTime = 0
        iterations = 10
        for i in range(iterations):
            plt.clf()
            plotBoard(ox, oy, sx, sy, gx, gy)
            startTime = time.time()
            planner = algorithm(
                ox, oy, resolution, robot_radius
            )  # Initialize the planner
            rx, ry = planner.plan(sx, sy, gx, gy, show_animation)
            finishTime = time.time()
            # planner.graph(rx, ry)
            cost = analyzeCost(rx, ry)
            totalCost += cost
            totalTime += finishTime - startTime
        print(
            f"| {algorithm.__name__} | {round(totalTime/iterations, 2)} | {round(totalCost/iterations, 2)} |"
        )
    # runRRtStar(obsList, [sx, sy], [gx, gy], robot_radius, show_animation)


def analyzeCost(rx, ry):
    cost = 0
    for i in range(len(rx) - 1):
        cost += math.hypot(rx[i] - rx[i + 1], ry[i] - ry[i + 1])
    return cost


def main():
    resolution = 1  # [m]
    robot_radius = 1  # [m]
    show_animation = True
    ox, oy, sx, sy, gx, gy = createGrid(True)
    obsList = [[ox[i], oy[i], 1] for i in range(len(ox))]
    runPlanners(
        ox, oy, sx, sy, gx, gy, robot_radius, show_animation, resolution, obsList
    )


if __name__ == "__main__":
    main()

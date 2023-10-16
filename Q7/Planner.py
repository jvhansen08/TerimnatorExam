import matplotlib.pyplot as plt


class Planner:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal

    def heuristic(self, node):
        # Implement a heuristic function for A*
        pass

    def step(self):
        # Implement how the algorithm takes a step
        pass

    def plan(self):
        # Implement the path planning logic
        pass

    def graph(self, path):
        # Visualize the path
        plt.imshow(self.grid, cmap="Greys", origin="upper")
        plt.plot(*zip(*path), marker="o", color="red", markersize=5, linewidth=2)
        plt.plot(self.start[1], self.start[0], marker="o", color="green", markersize=10)
        plt.plot(self.goal[1], self.goal[0], marker="o", color="blue", markersize=10)
        plt.show()

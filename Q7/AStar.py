from Dijkstra import Dijkstra
import matplotlib.pyplot as plt


class AStarPlanner(Dijkstra):
    def heuristic(self, node):
        # Implement the Manhattan distance heuristic for A*
        return abs(node.x - self.goal.x) + abs(node.y - self.goal.y)

    def step(self, current, neighbor):
        # A*: Use the heuristic to choose the next step
        return min(neighbor, key=lambda n: self.cost[n.x][n.y] + self.heuristic(n))

    def plan(self, sx, sy, gx, gy, show_animation=True):
        return super().planning(sx, sy, gx, gy, show_animation)

    def graph(self, rx, ry):
        plt.plot(rx, ry, "-r")
        plt.grid(True)
        plt.show()

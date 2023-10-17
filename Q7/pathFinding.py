import heapq


def createGrid():
    grid = [
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
        ],
        [1, 2, 2, 2, 2, 3, 1, 1, 1, 1],
        [1, 1, 1, -1, 3, 4, 3, 2, 2, 1],
        [2, 1, 1, -1, 4, 4, 4, 4, 2, 1],
        [2, 2, 1, -1, 4, 2, 4, 4, 3, 1],
        [3, 2, 2, -1, 2, 1, 1, 3, 3, 1],
        [3, 2, 2, -1, 2, 1, 1, 1, 2, 1],
        [4, 3, 2, 2, 2, 2, 1, 1, 1, 1],
    ]
    return grid


# Define the A* algorithm
def astar(grid, start, goal, moves):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {pos: float("inf") for row in grid for pos in row}
    g_score[start] = 0

    while open_set:
        current_cost, current_pos = heapq.heappop(open_set)

        if current_pos == goal:
            path = []
            while current_pos in came_from:
                path.append(current_pos)
                current_pos = came_from[current_pos]
            return path[::-1]

        for move_x, move_y in moves:
            new_x, new_y = current_pos[0] + move_x, current_pos[1] + move_y
            # Validate the new position is still on the grid and it is not an obstacle
            if (
                0 <= new_x < len(grid)
                and 0 <= new_y < len(grid[0])
                and grid[new_x][new_y] != -1
            ):
                tentative_g_score = g_score[current_pos] + 2
                cost = calculate_move_cost(new_x, new_y, current_pos, grid)
                tentative_g_score += cost
                if (new_x, new_y) not in g_score or tentative_g_score < g_score[
                    (new_x, new_y)
                ]:
                    came_from[(new_x, new_y)] = current_pos
                    g_score[(new_x, new_y)] = tentative_g_score
                    heapq.heappush(
                        open_set,
                        (
                            g_score[(new_x, new_y)] + heuristic((new_x, new_y), goal),
                            (new_x, new_y),
                        ),
                    )

    return None


# Define the heuristic function (Manhattan distance)
def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def calculate_move_cost(new_x, new_y, current_pos, grid):
    standardCost = 2
    if grid[new_x][new_y] > grid[current_pos[0]][current_pos[1]]:
        cost = standardCost + 1  # +1 cost for moving up in elevation
    elif grid[new_x][new_y] < grid[current_pos[0]][current_pos[1]]:
        cost = standardCost - 0.5
    else:  # Same elevation, so just standard cost
        cost = standardCost
    return cost


def calculate_path_cost(path, grid):
    totalCost = 0
    prevX, prevY = path[0]  # start at 0,0
    for x, y in path:
        totalCost += calculate_move_cost(x, y, (prevX, prevY), grid)
        prevX, prevY = x, y
    return totalCost


if __name__ == "__main__":
    start = (0, 0)
    goal = (6, 5)
    # Adjacency matrix
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    grid = createGrid()
    # Find the optimal path
    optimal_path = astar(grid, start, goal, moves)
    print(f"Optimal path: {optimal_path}")
    print(f"Path cost: {calculate_path_cost(optimal_path, grid)}")

    # Visualize the path on the grid
    if optimal_path:
        for pos in optimal_path:
            grid[pos[0]][pos[1]] = "P"  # Mark the path

    for row in reversed(grid):
        for cell in row:
            if cell == "P":
                print(f"{cell:^3}", end=" ")  # Adjust the width as needed
            else:
                print(f"{cell:3}", end=" ")
        print()

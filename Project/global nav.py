import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt as euc_dist

'''
res is cell size in milimeters


'''

def create_costmap(map, res, r_robot, margine, w, drop_rate):


    map_with_edge = np.pad(map, 1, constant_values=1)
    dists = euc_dist(map_with_edge==0)*res # free for 1, zero if not free
    dists = dists[1:-1, 1:-1]
    dists = np.maximum(dists - np.sqrt(2)*res, 0)

    safe_dist = r_robot + margine
    forbidden = dists <= safe_dist

    costs = 1.0 + w*np.exp(-np.maximum(dists - safe_dist, 0)/drop_rate)

    costs[forbidden] = np.inf
    return costs





def astar(costs, start, goal, res):

    def h(cell):
        x, y = cell
        return res * np.hypot(x - goal[0], y - goal[1])

    neighs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # check
    rows, cols = costs.shape
    for x, y in (start, goal):

        if not (0 <= y < cols and 0 <= x < rows):
            raise ValueError("goal outside of map")

        if not np.isfinite(costs[x, y]):
            raise ValueError("cannot go to obastacle")


    start = tuple(start)
    goal = tuple(goal)

    g = np.full((rows, cols), np.inf)
    g[start[0], start[1]] = 0.0
    parent = {}

    frontier = []
    heapq.heappush(frontier, (h(start), 0, start)) # (f, g, cell)

    while frontier:
        f, g_prev, curr = heapq.heappop(frontier)
        x,y = curr

        if g_prev > g[x, y]: continue

        if curr == goal:
            path = [curr]
            while curr != start:
                curr = parent[curr]
                path.append(curr)

            path.reverse()
            return path

        for (dx, dy) in neighs:
            new_row = x + dx
            new_col = y + dy

            if not (0 <= new_row < rows and 0 <= new_col < cols): continue
            if not np.isfinite(costs[new_row, new_col]): continue

            # dont allow diagonal movement except if both lateral cells are free
            if dx != 0 and dy != 0:
                if not np.isfinite(costs[x, new_col]) or not np.isfinite(costs[new_row, y]):
                    continue


            step_size = res * np.hypot(dx, dy)
            step_cost = (costs[new_row, new_col] + costs[x, y])/2 * step_size

            new_g = g[x, y] + step_cost

            if new_g < g[new_row, new_col]:

                neigh = (new_row, new_col)
                g[new_row, new_col] = new_g
                parent[neigh] = curr
                new_f = new_g + h(neigh)
                heapq.heappush(frontier, (new_f, new_g, neigh))


    return None # somehow


#continious position to cell indexes
def pos_to_cell(pos, res):

    x, y = pos

    xx = int(np.floor(y / res))
    yy = int(np.floor(x / res))
    new_pos = (xx, yy)

    return new_pos

#inverse
# return array of points robot center has to go in order to reach the goal
def path_cell_to_pos(path, res):

    if path is None:  return None

    return [((col + 0.5) * res, (row + 0.5) * res) for row, col in path]


def global_navigation(obstacle_map, robot_pos, goal_pos, res, r_robot, margine, w = 5.0, drop_rate = 100.0):

    costs = create_costmap(obstacle_map, res, r_robot, margine, w, drop_rate)

    start_cell = pos_to_cell(robot_pos, res)
    goal_cell = pos_to_cell(goal_pos, res)

    path = astar(costs, start_cell, goal_cell, res)

    if path is None: return None

    return path_cell_to_pos(path, res)


import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt as euc_dist


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

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


'''
we asume we know 
'''
















'''
def main2():

    # Parametri mape i planera.
    res = 0.025
    rows = 120
    cols = 160

    r_robot = 0.10
    margine = 0.04
    w = 4.0
    drop_rate = 0.10

    # Arena je 4 m x 3 m.
    width = cols * res
    height = rows * res

    start_position = (0.30, 0.30)
    goal_position = (3.70, 2.70)

    start_cell = pos_to_cell(start_position, res)
    goal_cell = pos_to_cell(goal_position, res)

    # Isti seed daje istu sekvencu mapa.
    # Promijeni 7 u drugi broj za drugačiju mapu.
    rng = np.random.default_rng(7)

    # Koordinate centara svih ćelija.
    xx, yy = np.meshgrid((np.arange(cols) + 0.5) * res, (np.arange(rows) + 0.5) * res)

    # Čuvamo slobodan prostor oko starta i cilja.
    start_zone = np.hypot(xx - start_position[0], yy - start_position[1]) < 0.35
    goal_zone = np.hypot(xx - goal_position[0], yy - goal_position[1]) < 0.35
    protected = start_zone | goal_zone

    path_cells = None

    # Ako je mapa neprohodna, pokušavamo sa novom.
    for attempt in range(100):

        obstacle_map = np.zeros((rows, cols), dtype=np.uint8)

        obstacle_count = 0

        for _ in range(1000):

            if obstacle_count == 25:
                break

            obstacle_rows = int(rng.integers(5, 17))
            obstacle_cols = int(rng.integers(5, 17))

            row = int(rng.integers(4, rows - obstacle_rows - 3))
            col = int(rng.integers(4, cols - obstacle_cols - 3))

            region = np.s_[row:row + obstacle_rows, col:col + obstacle_cols]

            if np.any(protected[region]):
                continue

            if np.any(obstacle_map[region]):
                continue

            obstacle_map[region] = 1
            obstacle_count += 1

        if obstacle_count < 25:
            continue

        costs = create_costmap(obstacle_map, res, r_robot, margine, w, drop_rate)

        try:
            path_cells = astar(costs, start_cell, goal_cell, res)

        except ValueError as error:
            print("Greška iz astar:", error)
            print("Dimenzije mape:", costs.shape)
            print("Start ćelija:", start_cell)
            print("Ciljna ćelija:", goal_cell)
            return

        if path_cells is None:
            print(f"Pokušaj {attempt + 1}: astar je vratio None.")

        if path_cells is not None:
            print(f"Prohodna mapa pronađena u pokušaju {attempt + 1}.")
            break

    if path_cells is None:
        print("Nije pronađena prohodna mapa nakon 100 pokušaja.")
        return

    # Putanja kroz centre ćelija, u metrima.
    path_world = path_cell_to_pos(path_cells, res)
    points = np.asarray(path_world, dtype=float)

    # Dužine segmenata i ukupna dužina putanje.
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    path_length = cumulative_lengths[-1]

    # Trošak putanje prema istoj formuli kao u A*.
    path_cost = 0.0

    for first, second in zip(path_cells[:-1], path_cells[1:]):

        row1, col1 = first
        row2, col2 = second

        step_length = res * np.hypot(row2 - row1, col2 - col1)
        path_cost += step_length * (costs[row1, col1] + costs[row2, col2]) / 2

    print(f"Number of waypointa: {len(points)}")
    print(f"Path lenght: {path_length:.2f} m")
    print(f"Overall cost: {path_cost:.3f}")

    # Priprema animacije pri približno konstantnoj brzini.
    speed = 0.20       # m/s u simulaciji
    fps = 30
    duration = path_length / speed

    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    travelled = np.linspace(0.0, path_length, frame_count)

    animation_x = np.interp(travelled, cumulative_lengths, points[:, 0])
    animation_y = np.interp(travelled, cumulative_lengths, points[:, 1])

    # Ugao prati smjer segmenta; nije simulacija upravljanja.
    headings = np.arctan2(np.diff(animation_y), np.diff(animation_x))
    headings = np.append(headings, headings[-1])

    # 0 = slobodno, 1 = sigurnosna zona, 2 = stvarna prepreka.
    display_map = np.zeros_like(obstacle_map)
    display_map[~np.isfinite(costs)] = 1
    display_map[obstacle_map == 1] = 2

    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = ListedColormap(["white", "#d0d0d0", "#252525"])

    ax.imshow(display_map, origin="lower", extent=[0, width, 0, height], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    # Cijela planirana putanja ostaje vidljiva.
    ax.plot(points[:, 0], points[:, 1], "--", color="royalblue", linewidth=1.8, label="A* path")

    ax.plot(points[0, 0], points[0, 1], "go", markersize=8, label="Start")
    ax.plot(points[-1, 0], points[-1, 1], "r*", markersize=14, label="Goal")

    # Robot je prikazan krugom koji ga obuhvata.
    robot = Circle(tuple(points[0]), r_robot, facecolor="orange", edgecolor="black", zorder=5, label="Robot")
    ax.add_patch(robot)

    # Trag već pređenog dijela putanje.
    trail, = ax.plot([], [], color="darkorange", linewidth=2.5, label="Path covered")

    # Linija koja pokazuje smjer robota.
    heading_line, = ax.plot([], [], "k-", linewidth=2, zorder=6)

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.9))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"A* navigation-length {path_length:.2f} m, cost {path_cost:.2f}")
    ax.legend(loc="lower right")

    def update(frame):

        x = animation_x[frame]
        y = animation_y[frame]
        theta = headings[frame]

        robot.center = (x, y)

        trail.set_data(animation_x[:frame + 1], animation_y[:frame + 1])

        tip_x = x + r_robot * np.cos(theta)
        tip_y = y + r_robot * np.sin(theta)

        heading_line.set_data([x, tip_x], [y, tip_y])

        if frame == frame_count - 1:
            status.set_text("Goal reached")
        else:
            status.set_text(f"distance covered: {travelled[frame]:.2f} / {path_length:.2f} m")

        return robot, trail, heading_line, status

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()
'''



def main():

    res = 0.025
    rows, cols = 120, 160

    r_robot = 0.10
    margine = 0.04
    w = 4.0
    drop_rate = 0.10

    speed = 0.20
    fps = 30

    start_position = (0.30, 1.50)
    goal_position = (3.70, 1.50)

    # Četiri prepreke, sa početno slobodnim srednjim prolazom.
    obstacle_map = np.zeros((rows, cols), dtype=np.uint8)

    obstacle_map[12:32, 30:46] = 1
    obstacle_map[85:105, 45:61] = 1
    obstacle_map[15:35, 100:116] = 1
    obstacle_map[88:108, 115:131] = 1

    def calculate_path(current_map, position):

        costs = create_costmap(current_map, res, r_robot, margine, w, drop_rate)

        start_cell = pos_to_cell(position, res)
        goal_cell = pos_to_cell(goal_position, res)

        cells = astar(costs, start_cell, goal_cell, res)

        if cells is None:
            return None, costs

        points = np.asarray(path_cell_to_pos(cells, res), dtype=float)

        # Dodaj stvarnu trenutnu poziciju ispred centra njene ćelije.
        if np.linalg.norm(points[0] - position) > 1e-9:
            points = np.vstack((position, points))

        return points, costs

    def sample_path(points):

        # Uklanjanje uzastopnih duplih tačaka.
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
        points = points[keep]

        distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]

        count = max(2, int(np.ceil(distances[-1] / speed * fps)) + 1)
        travelled = np.linspace(0.0, distances[-1], count)

        xs = np.interp(travelled, distances, points[:, 0])
        ys = np.interp(travelled, distances, points[:, 1])

        return np.column_stack((xs, ys))

    def display_grid(current_map, costs):

        display = np.zeros_like(current_map)
        display[~np.isfinite(costs)] = 1
        display[current_map == 1] = 2

        return display

    # 1. Početno planiranje.
    initial_path, initial_costs = calculate_path(obstacle_map, start_position)

    if initial_path is None:
        print("Nema početne putanje.")
        return

    initial_motion = sample_path(initial_path)

    # Robot staje nakon približno četvrtine početne putanje.
    stop_index = len(initial_motion) // 4
    stop_position = initial_motion[stop_index]

    # Nova prepreka se pojavljuje dalje ispred robota.
    obstacle_position = initial_motion[len(initial_motion) // 2]
    obstacle_row, obstacle_col = pos_to_cell(obstacle_position, res)

    changed_map = obstacle_map.copy()

    # Nova prepreka: približno 0.30 m široka i 0.80 m visoka.
    half_rows = round(0.40 / res)
    half_cols = round(0.15 / res)

    row_min = max(0, obstacle_row - half_rows)
    row_max = min(rows, obstacle_row + half_rows + 1)

    col_min = max(0, obstacle_col - half_cols)
    col_max = min(cols, obstacle_col + half_cols + 1)

    changed_map[row_min:row_max, col_min:col_max] = 1

    # 2. Replaniranje iz položaja u kojem se robot zaustavio.
    try:
        new_path, new_costs = calculate_path(changed_map, stop_position)
    except ValueError as error:
        print("Replaniranje nije moguće:", error)
        new_path = None
        new_costs = create_costmap(changed_map, res, r_robot, margine, w, drop_rate)

    # Animacija: kretanje → pauza → nova putanja.
    first_motion = initial_motion[:stop_index + 1]

    pause_frames = fps
    pause_motion = np.repeat(stop_position[None, :], pause_frames, axis=0)

    if new_path is not None:
        second_motion = sample_path(new_path)
        print("Pronađena je nova putanja oko dodate prepreke.")
    else:
        second_motion = np.repeat(stop_position[None, :], fps * 3, axis=0)
        print("Nema obilaska. Robot ostaje zaustavljen.")

    motion = np.vstack((first_motion, pause_motion, second_motion))

    change_frame = len(first_motion)
    resume_frame = change_frame + pause_frames

    # Orijentacija za crtanje.
    delta = np.diff(motion, axis=0)
    headings = np.zeros(len(motion))

    for i in range(len(delta)):
        if np.linalg.norm(delta[i]) > 1e-9:
            headings[i] = np.arctan2(delta[i, 1], delta[i, 0])
        elif i > 0:
            headings[i] = headings[i - 1]

    headings[-1] = headings[-2]

    # 3. Prikaz.
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = ListedColormap(["white", "#d0d0d0", "#252525"])

    map_image = ax.imshow(
        display_grid(obstacle_map, initial_costs),
        origin="lower",
        extent=[0, cols * res, 0, rows * res],
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )

    old_line, = ax.plot(initial_path[:, 0], initial_path[:, 1], "b--", linewidth=2, label="Početna A* putanja")
    new_line, = ax.plot([], [], color="green", linewidth=2, label="Nova A* putanja")
    trail, = ax.plot([], [], color="darkorange", linewidth=2, label="Pređeni put")

    ax.plot(*start_position, "go", markersize=8, label="Start")
    ax.plot(*goal_position, "r*", markersize=14, label="Cilj")

    robot = Circle(tuple(motion[0]), r_robot, facecolor="orange", edgecolor="black", zorder=5)
    ax.add_patch(robot)

    heading_line, = ax.plot([], [], "k-", linewidth=2, zorder=6)

    # Crveni okvir označava novododatu prepreku.
    obstacle_outline, = ax.plot([], [], "r-", linewidth=2, label="Nova prepreka")

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.95))

    ax.set_xlim(0, cols * res)
    ax.set_ylim(0, rows * res)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Promjena mape i globalno replaniranje")
    ax.legend(loc="lower right")

    def update(frame):

        x, y = motion[frame]
        theta = headings[frame]

        robot.center = (x, y)

        trail.set_data(motion[:frame + 1, 0], motion[:frame + 1, 1])
        heading_line.set_data([x, x + r_robot * np.cos(theta)], [y, y + r_robot * np.sin(theta)])

        if frame < change_frame:

            map_image.set_data(display_grid(obstacle_map, initial_costs))
            old_line.set_alpha(1.0)
            new_line.set_data([], [])
            obstacle_outline.set_data([], [])

            status.set_text("Praćenje početne putanje")

        else:

            map_image.set_data(display_grid(changed_map, new_costs))
            old_line.set_alpha(0.35)

            left = col_min * res
            right = col_max * res
            bottom = row_min * res
            top = row_max * res

            obstacle_outline.set_data([left, right, right, left, left], [bottom, bottom, top, top, bottom])

            if frame < resume_frame:

                new_line.set_data([], [])
                status.set_text("Nova prepreka na putanji — STOP i replaniranje")

            elif new_path is None:

                status.set_text("Nema nove putanje — robot ostaje zaustavljen")

            else:

                new_line.set_data(new_path[:, 0], new_path[:, 1])
                status.set_text("Praćenje nove putanje")

                if frame == len(motion) - 1:
                    status.set_text("Cilj dostignut")

        return map_image, robot, trail, heading_line, old_line, new_line, obstacle_outline, status

    animation = FuncAnimation(fig, update, frames=len(motion), interval=1000 / fps, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()











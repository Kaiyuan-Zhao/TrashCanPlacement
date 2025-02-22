import numpy
import pandas
import matplotlib.pyplot as plt
import heapq
import random
import pygame
import cv2
import numpy as np
import os

# Constants
NUM_AGENTS = 50
NUM_GARBAGE_CANS = 10

# Color constants
EMPTY = 0
WALL = 1
END = 3
SPAWN = 4

EMPTY_COLOR = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
END_COLOR = (255, 0, 0)
SPAWN_COLOR = (0, 0, 255)
CAN_COLOR = (0, 255, 0)

def load_map(file_path):
    return numpy.loadtxt(file_path, dtype=int)

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def pixToNum(pixel, colourMap, threshold=50):
    min_distance = float('inf')
    closest_color = -1
    for color, number in colourMap.items():
        distance = color_distance(pixel, color)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_color = number
    return closest_color

def translate(imPath, color_mapping, output_file):
    print(f"Reading image from path: {imPath}")
    if not os.path.exists(imPath):
        raise ValueError(f"Image file does not exist: {imPath}")

    image = cv2.imread(imPath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image read error")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    print("height: ", height, "width: ", width)

    with open(output_file, "w") as f:
        for y in range(height):
            row = []
            for x in range(width):
                row.append(str(pixToNum(image[y, x], color_mapping)))
            f.write(" ".join(row) + "\n")

class Agent:
    def __init__(self, start, end, patience, sight, map_data):
        self.x, self.y = start
        self.patience = patience
        self.end = end
        self.sight = sight
        self.map_data = map_data
        self.path = self.a_star_pathfind(start, self.end)  # precompute path
        self.has_littered = False
        self.active = True

    def a_star_pathfind(self, start, end):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy, cost in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.map_data.shape[0] and 0 <= neighbor[1] < self.map_data.shape[1]:
                    if self.map_data[neighbor[0], neighbor[1]] == WALL:
                        continue
                    tentative_g_score = g_score[current] + cost
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def is_visible(self, target):
        x0, y0 = self.x, self.y
        x1, y1 = target
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        if dx > self.sight or dy > self.sight:
            return False

        while (x0, y0) != (x1, y1):
            if self.map_data[x0, y0] == WALL:
                return False
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def update(self, garbage_cans):
        if (self.x, self.y) == self.end:
            self.active = False
            return
        if self.patience <= 0:
            self.has_littered = True
            self.active = False
            return
        if not self.path:
            self.active = False
            return

        closest_can = None
        closest_distance = float('inf')
        for can in garbage_cans:
            if self.is_visible(can):
                distance = abs(self.x - can[0]) + abs(self.y - can[1])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_can = can

        if closest_can:
            self.path = self.a_star_pathfind((self.x, self.y), closest_can)

        try:
            self.x, self.y = self.path.pop(0)
        except IndexError:
            return True
        self.patience -= 1

def run_simulation(map_data, agents, garbage_cans, visualize=False, tick_speed=600000):
    rows, cols = map_data.shape
    SCREEN_WIDTH = 1024  # Desired window width
    SCREEN_HEIGHT = 1024  # Desired window height
    TILE_SIZE = min(SCREEN_WIDTH // cols, SCREEN_HEIGHT // rows)  # Calculate tile size to fit the window

    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()

    garbage_dropped = 0
    running = True
    while agents and running:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    running = False

        for agent in agents:
            if agent.active:
                agent.update(garbage_cans)

        if not agents:
            break  # Terminate the simulation if all agents are immobilized or removed

        if visualize:
            screen.fill(EMPTY_COLOR)
            # Draw map
            for row in range(rows):
                for col in range(cols):
                    if map_data[row, col] == WALL:
                        color = WALL_COLOR 
                    elif map_data[row, col] == END:
                        color = END_COLOR 
                    elif map_data[row, col] == SPAWN:
                        color = SPAWN_COLOR
                    else:
                        color = EMPTY_COLOR
                    pygame.draw.rect(
                        screen,
                        color,
                        (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    )

            # Draw garbage cans
            for can in garbage_cans:
                pygame.draw.rect(
                    screen,
                    CAN_COLOR,
                    (can[1] * TILE_SIZE, can[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )

            # Draw agents and their sight
            for agent in agents:
                pygame.draw.circle(
                    screen,
                    CAN_COLOR,
                    (agent.y * TILE_SIZE + TILE_SIZE // 2, agent.x * TILE_SIZE + TILE_SIZE // 2),
                    agent.sight * TILE_SIZE,
                    1
                )
                pygame.draw.rect(
                    screen,
                    SPAWN_COLOR,
                    (agent.y * TILE_SIZE, agent.x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )
                if agent.path:
                    path_points = [(p[1] * TILE_SIZE + TILE_SIZE // 2, p[0] * TILE_SIZE + TILE_SIZE // 2) for p in agent.path]
                    try:
                        pygame.draw.lines(screen, (0, 255, 255), False, path_points, 2)
                    except:
                        pass
            if all(not agent.active for agent in agents):
                running = False
                break
            pygame.display.flip()
            clock.tick(tick_speed)

    for agent in agents:
        if agent.has_littered:
            garbage_dropped += 1
    if visualize:
        print("Simulation ended. Press 'c' to close the window.")
        running = False
    
    print("Garbage dropped:", garbage_dropped)
    return garbage_dropped

if __name__ == '__main__':
    # Convert map image to array
    imPath = "map2.jpg"  # Change this to the path of your image
    output = "output.txt"  # Output teçxt file

    # Colours
    colours = {
        EMPTY_COLOR: EMPTY,
        WALL_COLOR: WALL,  
        SPAWN_COLOR: SPAWN,
        END_COLOR: END  
    }

    translate(imPath, colours, output)

    # Load map data from output file
    map_data = load_map(output)
    print("Map data shape:", map_data.shape)
    print(map_data)
    MAP_SIZE = map_data.shape

    # Prepare heatmap array
    heatmap = np.zeros(MAP_SIZE, dtype=int)

    NUM_RUNS = 10
    for run in range(NUM_RUNS):
        # Reset seed for identical simulation runs
        garbage_cans = []
        while len(garbage_cans) < NUM_GARBAGE_CANS:
            can = (random.randint(0, MAP_SIZE[0] - 1), random.randint(0, MAP_SIZE[1] - 1))
            if map_data[can[0], can[1]] != WALL:
                adjacent_to_wall = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (can[0] + dx, can[1] + dy)
                    if 0 <= neighbor[0] < MAP_SIZE[0] and 0 <= neighbor[1] < MAP_SIZE[1]:
                        if map_data[neighbor[0], neighbor[1]] == WALL:
                            adjacent_to_wall = True
                            break
                if adjacent_to_wall and map_data[can[0], can[1]] == EMPTY:
                    garbage_cans.append(can)
        
        # Find all destination points (marked with END value)
        destination_points = [(i, j) for i in range(MAP_SIZE[0]) for j in range(MAP_SIZE[1]) if map_data[i, j] == END]
        if not destination_points:
            raise ValueError("No destination points found in the map data.")

        # Create agents
        agents = []
        spawn_points = [(i, j) for i in range(MAP_SIZE[0]) for j in range(MAP_SIZE[1]) if map_data[i, j] == SPAWN]
        for _ in range(NUM_AGENTS):
            start = random.choice(spawn_points)
            end = random.choice(destination_points)
            agents.append(Agent(start=start, end=end, patience=random.randint(100, 300), sight=random.randint(5, 15), map_data=map_data))

        # Run simulation without visualization
        gd = run_simulation(map_data, agents, garbage_cans, visualize=True)

        # Update heatmap for each trashcan location with (NUM_AGENTS - garbage dropped)
        for can in garbage_cans:
            heatmap[can[0], can[1]] += (NUM_AGENTS - gd)

    # Display heatmap using matplotlib with intensity based on final value at each cell
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("Trashcan Heatmap Over {} Runs".format(NUM_RUNS))
    plt.colorbar(label="Accumulated Score")
    plt.show()
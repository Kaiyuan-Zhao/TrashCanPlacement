import numpy
#import pandas
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
NUM_RUNS = 500
maxPercentage = 60  # Maximum percentage scale for heatmap visualization
random_spawn_dest = True # Set to True to enable random spawn/destination
visualize = False
imPath = "Map3.png"  # Change this to the path of your image

paddingMultiplyer = 1

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
            can_pos = can.get_position()
            if self.is_visible(can_pos):
                distance = abs(self.x - can_pos[0]) + abs(self.y - can_pos[1])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_can = can

        try:
            self.x, self.y = self.path.pop(0)
        except IndexError:
            return True

        if closest_can:
            self.path = self.a_star_pathfind((self.x, self.y), closest_can.get_position())
            # Record visit when agent reaches the can (position matches can position)
            if (self.x, self.y) == closest_can.get_position():
                closest_can.record_visit(not self.has_littered)
                #print(f"Agent visited can at ({self.x},{self.y}) - success: {not self.has_littered}")
        self.patience -= 1

class GarbageCan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.successful_visits = 0  # Counts only successful disposals
    
    def get_position(self):
        return (self.x, self.y)
    
    def record_visit(self, success):
        if success:
            self.successful_visits += 1
    
    def get_success_count(self):
        return self.successful_visits

def run_simulation(map_data, agents, garbage_cans, visualize=False, tick_speed=60):
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
    end_simulation = False  # Flag to end simulation

    while agents and running:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    running = False
                    end_simulation = True  # Set flag to end simulation

        if not visualize:
            try:
                import msvcrt  # Windows-specific module
                if msvcrt.kbhit():  # Check if key pressed
                    if msvcrt.getch() == b'c':  # Check if 'c' was pressed
                        running = False
                        end_simulation = True
            except ImportError:
                # Fallback for non-Windows systems
                import sys
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.read(1)
                    if line == 'c':
                        running = False
                        end_simulation = True

        if end_simulation:
            break  # Break out of the loop if 'c' is pressed

        for agent in agents:
            if agent.active:
                agent.update(garbage_cans)

        # Check if all agents are inactive
        if all(not agent.active for agent in agents):
            break

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
                    (can.y * TILE_SIZE, can.x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
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

            pygame.display.flip()
            clock.tick(tick_speed)

    for agent in agents:
        if agent.has_littered:
            garbage_dropped += 1
    if visualize:
        print("Simulation ended. Press 'c' to close the window.")
        pygame.quit()

    print("Garbage dropped:", garbage_dropped)
    return garbage_dropped

def run_combined_simulation(map_data, agents, garbage_cans, visualize=False, tick_speed=60):
    rows, cols = map_data.shape
    path_heatmap = np.zeros((rows, cols), dtype=int)
    trash_heatmap = np.zeros((rows, cols), dtype=int)
    count_map = np.zeros((rows, cols), dtype=int)
    
    # First pass: Calculate path coverage with radius
    for agent in agents:
        radius = agent.sight
        for x, y in agent.path:
            # Mark all tiles within radius of path points
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < rows and 0 <= ny < cols and 
                        map_data[nx, ny] != WALL and 
                        dx*dx + dy*dy <= radius*radius):
                        path_heatmap[nx, ny] += 1
    
    # Second pass: Run garbage collection simulation
    garbage_dropped = run_simulation(map_data, agents, garbage_cans, visualize=visualize, tick_speed=tick_speed)
    #!garbage dropped should not be used now

    #print("------------------")
    # Calculate percentage of successful disposals per agent for this run
    for can in garbage_cans:
        #print(can.successful_visits)
        percentage = (can.successful_visits / NUM_AGENTS) * 100
        trash_heatmap[can.x, can.y] += percentage
    #print("------------------")
    
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Path coverage heatmap
        im1 = ax1.imshow(path_heatmap, cmap='hot', interpolation='nearest')
        ax1.set_title("Agent Path Coverage (Radius = Sight)")
        fig.colorbar(im1, ax=ax1, label="Coverage Count")
        
        # Average the percentages across runs
        avg_trash = np.divide(trash_heatmap, NUM_RUNS,
                            out=np.zeros_like(trash_heatmap, dtype=float))
        masked_avg = np.ma.masked_where(trash_heatmap == 0, avg_trash)
        im2 = ax2.imshow(masked_avg, cmap='hot', interpolation='nearest',
                        norm=plt.Normalize(vmin=0, vmax=maxPercentage))
        ax2.set_title("Trash Collection Rate (%)")
        fig.colorbar(im2, ax=ax2, label="Percentage")
        
        plt.tight_layout()
        plt.show()
    
    return path_heatmap, trash_heatmap

if __name__ == '__main__':
    # Convert map image to array
    
    output = "output.txt"  # Output text file

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

    # Define scaling factor based on map size
    SCALE_FACTOR = (MAP_SIZE[0] / 100 + MAP_SIZE[1] / 100) / 2
    NUM_GARBAGE_CANS = int(NUM_GARBAGE_CANS * SCALE_FACTOR)
    NUM_AGENTS = int(NUM_AGENTS * SCALE_FACTOR)
    print("Scale factor:", SCALE_FACTOR)
    print("Number of garbage cans:", NUM_GARBAGE_CANS)
    print("Number of agents:", NUM_AGENTS)

    '''
    # Original trash simulation code (commented out)
    NUM_RUNS = 500
    for run in range(NUM_RUNS):
        # Reset seed for identical simulation runs
        garbage_cans = []
        while len(garbage_cans) < NUM_GARBAGE_CANS:
            x = random.randint(0, MAP_SIZE[0] - 1)
            y = random.randint(0, MAP_SIZE[1] - 1)
            if map_data[x, y] != WALL:
                adjacent_to_wall = False
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < MAP_SIZE[0] and 0 <= neighbor[1] < MAP_SIZE[1]:
                        if map_data[neighbor[0], neighbor[1]] == WALL:
                            adjacent_to_wall = True
                            break
                if adjacent_to_wall and (map_data[x, y] == EMPTY or map_data[x, y] == END):
                    garbage_cans.append(GarbageCan(x, y))
        
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
            patience = int(random.randint(25, 75) * SCALE_FACTOR)
            sight = int(random.randint(4, 8) * SCALE_FACTOR)
            agents.append(Agent(start=start, end=end, patience=patience, sight=sight, map_data=map_data))

        # Run simulation without visualization
        gd = run_simulation(map_data, agents, garbage_cans, visualize=False)

        # Update heatmap and count_map for each trashcan location
        for can in garbage_cans:
            heatmap[can[0], can[1]] += (NUM_AGENTS - gd)
            count_map[can.x, can.y] += 1
        print("Run", run + 1, "completed.")
    # Calculate average percentage of trash not dropped
    average_heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap, dtype=float), where=count_map != 0)

    # Display heatmap using matplotlib with intensity based on final value at each cell
    plt.figure(figsize=(8, 8))
    plt.imshow(average_heatmap, cmap='hot', interpolation='nearest', norm=plt.Normalize(vmin=0, vmax=np.max(average_heatmap) * 1.2))
    plt.title("Average Percentage of Trash Not Dropped Over {} Runs".format(NUM_RUNS))
    plt.colorbar(label="Average Percentage")
    plt.show()
    '''


    # Combined simulation showing both path coverage and trash collection
    # Set random_spawn_dest=True to allow agents to spawn and go to any non-wall tile

    path_heatmap = np.zeros(MAP_SIZE, dtype=int)
    trash_heatmap = np.zeros(MAP_SIZE, dtype=int)
    count_map = np.zeros(MAP_SIZE, dtype=int)
    
    for run in range(NUM_RUNS):
        # Reset seed for identical simulation runs
        garbage_cans = []
        while len(garbage_cans) < NUM_GARBAGE_CANS:
            x = random.randint(0, MAP_SIZE[0] - 1)
            y = random.randint(0, MAP_SIZE[1] - 1)
            if map_data[x, y] != WALL:
                adjacent_to_wall = False
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < MAP_SIZE[0] and 0 <= neighbor[1] < MAP_SIZE[1]:
                        if map_data[neighbor[0], neighbor[1]] == WALL:
                            adjacent_to_wall = True
                            break
                if adjacent_to_wall and (map_data[x, y] == EMPTY or map_data[x, y] == END):
                    garbage_cans.append(GarbageCan(x, y))
        
        # Find all possible spawn and destination points
        if random_spawn_dest:
            # Only END (red) and SPAWN (blue) squares can be spawn/destination
            valid_points = [(i, j) for i in range(MAP_SIZE[0]) for j in range(MAP_SIZE[1]) 
                          if map_data[i, j] in (END, SPAWN)]
            spawn_points = valid_points.copy()
            destination_points = valid_points.copy()
        else:
            # Use original SPAWN and END points
            spawn_points = [(i, j) for i in range(MAP_SIZE[0]) for j in range(MAP_SIZE[1]) 
                          if map_data[i, j] == SPAWN]
            destination_points = [(i, j) for i in range(MAP_SIZE[0]) for j in range(MAP_SIZE[1]) 
                                if map_data[i, j] == END]
        
        if not spawn_points:
            raise ValueError("No spawn points found in the map data.")
        if not destination_points and not random_spawn_dest:
            raise ValueError("No destination points found in the map data.")

        # Create agents
        agents = []
        for _ in range(NUM_AGENTS):
            start = random.choice(spawn_points)
            end = random.choice(destination_points)
            patience = int(random.randint(25, 75) * SCALE_FACTOR)
            sight = int(random.randint(4, 8) * SCALE_FACTOR)
            agents.append(Agent(start=start, end=end, patience=patience, sight=sight, map_data=map_data))

        # Run combined simulation with random spawn/dest toggle
        run_path_heatmap, run_trash_heatmap = run_combined_simulation(
            map_data, agents, garbage_cans, 
            visualize=visualize,)
        path_heatmap += run_path_heatmap
        trash_heatmap += run_trash_heatmap
        
        # Update count map for trash cans
        for can in garbage_cans:
            count_map[can.x, can.y] += 1
        
        print("Run", run + 1, "completed.")

    # Calculate average trash collection percentages
    avg_trash = np.divide(trash_heatmap, count_map, 
                         out=np.zeros_like(trash_heatmap, dtype=float), 
                         where=count_map != 0)

    # Calculate average path coverage
    avg_path = np.divide(path_heatmap, NUM_RUNS,
                        out=np.zeros_like(path_heatmap, dtype=float),
                        where=path_heatmap!=0)

    # Display final averaged heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Path coverage heatmap (averaged)
    im1 = ax1.imshow(avg_path, cmap='hot', interpolation='nearest',
                    norm=plt.Normalize(vmin=0, vmax=np.max(avg_path)*paddingMultiplyer))
    ax1.set_title("Average Path Coverage Over {} Runs".format(NUM_RUNS))
    fig.colorbar(im1, ax=ax1, label="Average Coverage")
    
    # Trash collection heatmap (averaged)
    im2 = ax2.imshow(avg_trash, cmap='hot', interpolation='nearest', norm=plt.Normalize(vmin=0, vmax=maxPercentage*paddingMultiplyer))
    ax2.set_title("Average Trash Collected Over {} Runs".format(NUM_RUNS))
    fig.colorbar(im2, ax=ax2, label="Average Percentage")

    plt.tight_layout()
    plt.show()

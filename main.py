import numpy
import pandas
import matplotlib.pyplot as plt
import heapq
import random
import pygame

SCREEN_WIDTH = 1024  # Separate constant for screen width
SCREEN_HEIGHT = 1024  # Separate constant for screen height
MAP_SIZE = 256  # Size of the map (256x256)
TILE_SIZE = SCREEN_WIDTH // MAP_SIZE  # Calculate tile size to maximize screen usage

class Agent:
    def __init__(self, start, end, patience, sight, map_data):
        self.x, self.y = start
        self.end = end
        self.patience = patience
        self.sight = sight
        self.map_data = map_data
        self.path = self.a_star_pathfind(start, end)  # precompute path
        self.original_end = end
        self.has_dropped_garbage = False
    
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

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.map_data.shape[0] and 0 <= neighbor[1] < self.map_data.shape[1]:
                    if self.map_data[neighbor[0], neighbor[1]] == 1:
                        continue
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def update(self, garbage_cans):
        if not self.path or self.patience <= 0:
            self.has_dropped_garbage = True
            return False
        
        # Check for garbage cans within sight
        for can in garbage_cans:
            if abs(self.x - can[0]) <= self.sight and abs(self.y - can[1]) <= self.sight:
                self.path = self.a_star_pathfind((self.x, self.y), can)
                self.end = self.original_end
        try:
            self.x, self.y = self.path.pop(0)
        except IndexError:
            self.has_dropped_garbage = True
            return False
        self.patience -= 1
        return (self.x, self.y) != self.end

def run_simulation(map_data, agents, garbage_cans, visualize=False, tick_speed=60):
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()

    garbage_dropped = 0
    while agents:
        if visualize:
            pygame.event.pump()
        remaining_agents = []
        for agent in agents:
            active = agent.update(garbage_cans)
            if not active and agent.has_dropped_garbage:
                if (agent.x, agent.y) not in garbage_cans:
                    garbage_dropped += 1
            else:
                remaining_agents.append(agent)
        agents = remaining_agents

        if visualize:
            screen.fill((255, 255, 255))
            # Draw map
            rows, cols = map_data.shape
            for row in range(rows):
                for col in range(cols):
                    if map_data[row, col] == 1:
                        color = (0, 0, 0)  # Wall
                    elif map_data[row, col] == 2:
                        color = (128, 0, 128)  # Start point
                    elif map_data[row, col] == 4:
                        color = (255, 0, 255)  # End point
                    else:
                        color = (255, 255, 255)  # Walkable
                    pygame.draw.rect(
                        screen,
                        color,
                        (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    )

            # Draw garbage cans
            for can in garbage_cans:
                pygame.draw.rect(
                    screen,
                    (255, 0, 0),
                    (can[1] * TILE_SIZE, can[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )

            # Draw agents and their sight
            for agent in agents:
                # Sight circle
                pygame.draw.circle(
                    screen,
                    (0, 255, 0),
                    (agent.y * TILE_SIZE + TILE_SIZE // 2, agent.x * TILE_SIZE + TILE_SIZE // 2),
                    agent.sight * TILE_SIZE,
                    1
                )
                # Agent as a TILE_SIZE x TILE_SIZE block
                pygame.draw.rect(
                    screen,
                    (0, 0, 255),
                    (agent.y * TILE_SIZE, agent.x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )
            pygame.display.flip()
            clock.tick(tick_speed)

    if visualize:
        pygame.quit()

    print("Garbage dropped:", garbage_dropped)

# Placeholder map data
map_data = numpy.zeros((MAP_SIZE, MAP_SIZE))
map_data[50:80, 100:180] = 1
map_data[150:170, 250:256] = 1
map_data[50, 50] = 2
map_data[200, 200] = 4

# Create agents
agents = [
    Agent(
        start=(random.randint(0, MAP_SIZE // 4), random.randint(0, MAP_SIZE // 4)),
        end=(random.randint(3 * MAP_SIZE // 4, MAP_SIZE - 1), random.randint(3 * MAP_SIZE // 4, MAP_SIZE - 1)),
        patience=random.randint(100, 300),
        sight=random.randint(5, 15),
        map_data=map_data
    )
    for _ in range(100)
]

# Create garbage cans
garbage_cans = [
    (random.randint(0, MAP_SIZE - 1), random.randint(0, MAP_SIZE - 1))
    for _ in range(20)
]

# Run simulation
run_simulation(map_data, agents, garbage_cans, visualize=True, tick_speed=30)
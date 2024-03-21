import pygame
import random
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 150
ELITISM_RATE = 0.2
MUTATION_RATE = 0.1
MAX_GENERATIONS = 10

# --- Environment Parameters ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

upper_bound = (0, 0)
lower_bound = (WINDOW_WIDTH, WINDOW_HEIGHT)
alpha = (math.sqrt((lower_bound[0] - upper_bound[0]) ** 2 + (lower_bound[1] - upper_bound[1]) ** 2)) * 0.01

OBSTACLE_WIDTH = 40
OBSTACLE_COLOR = (255, 0, 0)
obstacles = [(300, 300), (200, 200), (400, 400), (500, 200), (600, 300)]

POSITION = [(100, 100), (100, 500), (700, 500), (700, 100)]
END_POSITION = ()
START_POSITION = ()
s = 0  # int(input("Enter the start-node in 0-3:"))
e = 2  # int(input("Enter the end-node in 0-3:"))
START_POSITION = START_POSITION + tuple(POSITION[s])
END_POSITION = END_POSITION + tuple(POSITION[e])
START_POSITION = tuple(START_POSITION)
END_POSITION = tuple(END_POSITION)

START_COLOR = (0, 255, 0)
END_COLOR = (0, 0, 255)

fes_path = []
# point_coord=[]
overall_best_path_dist = []  # contains all the best path dist of all generations. from this we can select the overall best path out of all the generations
mutation_coord = [[] for i in range(MAX_GENERATIONS)]
temp_mutation_coord = [[] for i in range(MAX_GENERATIONS)]


# --- Helper Functions ---
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def check_collision(point):
    for obstacle in obstacles:
        if calculate_distance(point, obstacle) < OBSTACLE_WIDTH / 2 + 10:
            return True
    return False


def generate_population(start, end, num_points=7):
    population = []
    f_path = 0
    for i in range(POPULATION_SIZE):
        path = [start]
        for j in range(num_points - 2):  # Generate paths with 7 points
            point = (random.randint(0, WINDOW_WIDTH - 100), random.randint(0, WINDOW_HEIGHT - 100))

            # point_coord.append(point)

            while check_collision(point):
                point = (random.randint(0, WINDOW_WIDTH - 100), random.randint(0, WINDOW_HEIGHT - 100))
            path.append(point)
        path.append(end)
        population.append(path)
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]
        if collision(start_point, end_point) != 2 and 1:
            f_path = f_path+1
    print("Feasible paths",f_path)
    return population


def fitness(path):
    total_distance = 0
    # path_length = sum([calculate_distance(path[i], path[i + 1]) for i in range(len(path) - 1)])
    # print(f)
    for i in range(len(path) - 1):
        path_length = sum([calculate_distance(path[i], path[i + 1]) for i in range(len(path) - 1)])
        if path_length == 0:
            smooth = 0
        else:
            smooth = sum([d(path[i], path[i + 1], path[i + 2]) for i in range(len(path) - 2)]) / path_length
        total_distance += path_length + smooth
    fit = total_distance
    return (1 / fit)


def crossover(child1, child2):
    child = [child1[0]]
    for i in range(1, len(child1) - 1):
        if random.random() < 0.5:
            child.append(child1[i])
        else:
            child.append(child2[i])
    child.append(child1[-1])
    return child


def highlight_mutation_point(temp_mutation_coord, surface, gen_index):
    for coord in temp_mutation_coord[gen_index]:
        pygame.draw.circle(surface, (100, 150, 100), coord, 8)


def find_mutate(best_path, surface,
                gen_index):  # try to find mutation points for the best path alone and save in temp_mutation_coord
    for coord in mutation_coord[gen_index]:
        if coord in best_path:
            temp_mutation_coord[gen_index].append(coord)

        highlight_mutation_point(temp_mutation_coord, surface, gen_index)


def mutate(path, gen_index):
    if gen_index < MAX_GENERATIONS:
        for j in range(1, len(path) - 1):
            if random.random() < MUTATION_RATE:
                point = (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT))
                while check_collision(point):
                    point = (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT))

                mutation_coord[gen_index].append(point)
                path[j] = point

    return path


def select_parents(population):
    population_size = len(population)
    fitness_scores = [fitness(path) for path in population]
    total_fitness = sum(fitness_scores)
    probabilities = [fitness_scores[i] / total_fitness for i in range(population_size)]
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(population_size)]
    elite_size = int(ELITISM_RATE * population_size)
    elite_indices = sorted(range(population_size), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
    non_elite_indices = [i for i in range(population_size) if i not in elite_indices]
    parents = [population[i] for i in elite_indices]
    for i in range(int((1 - ELITISM_RATE) * population_size)):
        r = random.random()
        for j in range(population_size - elite_size):
            if r <= cumulative_probabilities[non_elite_indices[j]]:
                parents.append(population[non_elite_indices[j]])
                break
    return parents


def evolve_population(population, gen_index):
    new_population = []
    elitism_size = int(ELITISM_RATE * len(population))
    elite_paths = sorted(population, key=fitness, reverse=True)[:elitism_size]
    new_population.extend(elite_paths)
    while len(new_population) < len(population):
        parents = select_parents(population)
        child1, child2 = parents[0], parents[1]
        child = crossover(child1, child2)
        child = mutate(child, gen_index)
        if not check_collision(child[-2]):
            new_population.append(child)
    return new_population


def d(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    distance12 = calculate_distance(point1, point2)
    distance13 = calculate_distance(point1, point3)
    if distance12 == 0 or distance13 == 0:
        return 1  # or any other value that makes sense in your context
    value = ((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)) / (distance12 * distance13)
    value = np.clip(value, -1, 1)
    a = 3.14 - np.arccos(value)
    # print(a)
    return a


def final_feasible_path(best_path, i):
    # Calculate fitness of best path
    start_time = time.time()
    best_fitness = fitness(best_path)
    best_path_length = sum([calculate_distance(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)])
    # best_path_time = best_path_length / best_fitness
    best_smoothness = sum(
        [d(best_path[i], best_path[i + 1], best_path[i + 2]) for i in range(len(best_path) - 2)]) / best_path_length
    end_time = time.time()
    # Print fitness of best path in generation
    print(
        f"\n\nGeneration {i + 1}: Best fitness = {best_fitness:.4f}, Path length = {best_path_length:.4f},Smoothness = {best_smoothness:.4f},Time taken for the path = {(end_time - start_time):.4f}")
    pygame.display.update()

    overall_best_path_dist.append(best_path_length)


def draw_path_points(path, surface):  # printing dots of the nodes in the path
    x1, y1 = START_POSITION
    x2, y2 = END_POSITION
    for point in path:
        if point[0] == x1 and point[1] == y1:
            pygame.draw.circle(surface, (0, 0, 255), point, 1)
        elif point[0] == x2 and point[1] == y2:
            pygame.draw.circle(surface, (0, 255, 0), point, 1)
        else:
            pygame.draw.circle(surface, (255, 0, 0), point, 4)


obstacle_rects = []


def draw_obstacles(surface):  # draws the obstacles
    for obstacle in obstacles:
        x, y = obstacle
        rect = pygame.Rect(x - OBSTACLE_WIDTH / 2, y - OBSTACLE_WIDTH / 2, OBSTACLE_WIDTH, OBSTACLE_WIDTH)
        pygame.draw.rect(surface, OBSTACLE_COLOR, rect)
        obstacle_rects.append(rect)


def collision(start, end):
    for obstacle in obstacle_rects:
        if obstacle.collidepoint(start) or obstacle.collidepoint(end):
            return 1
        elif obstacle.clipline(start, end):
            return 2


infes_path = []
l1 = []


def draw_path(path, surface):  # draws path lines along with the coordinatesfor the nodes
    font = pygame.font.SysFont("Calibri (Body)", 20)
    pygame.draw.lines(surface, (0, 0, 0), False, path, 2)
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]
        if collision(start_point, end_point) != 2 and 1:
            pygame.draw.line(surface, (0, 0, 0), start_point, end_point, 5)
        else:
            print("infeasible path detected")
            l = 1
            l1.append(l)
            infes_path.append(path)
            #print(infes_path)
            #print(l1)
            break

    x1, y1 = START_POSITION
    x2, y2 = END_POSITION
    for point in path:
        if point[0] == x1 and point[1] == y1:
            text = font.render('', True, (0, 0, 0))
        elif point[0] == x2 and point[1] == y2:
            text = font.render('', True, (0, 0, 0))
        else:
            text = font.render(f"({point[0]}, {point[1]})", True, (0, 0, 0))
            surface.blit(text, point)


def draw_start_and_end(surface):  # circle for start and end
    pygame.draw.circle(surface, START_COLOR, START_POSITION, 14)
    pygame.draw.circle(surface, END_COLOR, END_POSITION, 14)


def label_obstacles(surface):  # label o for obstacles
    font = pygame.font.Font(None, 25)
    for i, obstacle in enumerate(obstacles):
        x, y = obstacle
        label = font.render('O', True, (0, 0, 0))
        label_rect = label.get_rect(center=obstacle)
        surface.blit(label, label_rect)


def label_start_end(surface):  # giving a label for start and end
    font = pygame.font.SysFont("Arial", 16)

    label_start = font.render('START', True, (0, 0, 0))
    label_start_rect = label_start.get_rect(midtop=(100, 60))
    surface.blit(label_start, label_start_rect)

    label_end = font.render('END', True, (0, 0, 0))
    label_end_rect = label_end.get_rect(midtop=(700, 460))
    surface.blit(label_end, label_end_rect)


def label_path_distance(surface, path):  # printing the distance for the final path for every generation
    total_distance = 0
    font = pygame.font.SysFont("Arial", 17)
    label_dist = font.render('DISTANCE : ', True, (0, 0, 0))
    label_dist_rect = label_dist.get_rect(midtop=(622, 40))
    surface.blit(label_dist, label_dist_rect)

    for i in range(len(path) - 1):
        total_distance += calculate_distance(path[i], path[i + 1])
        total_distance = round(total_distance, 4)
        label_dist_val = font.render(str(total_distance), True, (0, 0, 0))
        label_dist_val_rect = label_dist.get_rect(midtop=(700, 40))
    surface.blit(label_dist_val, label_dist_val_rect)


def dd(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist > alpha:
        return alpha
    return dist


def genetic_algorithm(population_size, num_generations, crossover_probability, mutation_probability):
    # Initialize the population

    population = np.random.randint(10, size=(population_size, 10, 2))
    best_fitness_values = []
    avg_fitness_values = []
    # Evolve the population
    for generation in range(num_generations):
        # Calculate the fitness of each individual
        fitness_values = np.array([fitness(individual) for individual in population])
        best_fitness = np.max(fitness_values)
        best_fitness_values.append(best_fitness)
        avg_fitness = np.mean(fitness_values)
        avg_fitness_values.append(avg_fitness)
        # Select the parents for crossover
        parents = population[np.random.choice(population_size, size=population_size - 1, replace=True,
                                              p=fitness_values / fitness_values.sum())]
        # Perform crossover
        children = []
        for i in range(0, population_size - 1, 2):
            if np.random.rand() < crossover_probability:
                crossover_point = np.random.randint(1, 9)
                child1 = np.concatenate((parents[i, :crossover_point, :], parents[i + 1, crossover_point:, :]), axis=0)
                child2 = np.concatenate((parents[i + 1, :crossover_point, :], parents[i, crossover_point:, :]), axis=0)
            else:
                child1 = parents[i]
                child2 = parents[i + 1]
            children.append(child1)
            children.append(child2)
        children = np.array(children)
        # Perform mutation
        for i in range(population_size - 1):
            if np.random.rand() < mutation_probability:
                mutation_point = np.random.randint(10)
                children[i, mutation_point, :] = np.random.randint(10, size=2)
        # Replace the least fit individual with the most fit individual from the previous generation
        population[np.argmin(fitness_values)] = population[np.argmax(fitness_values)]
        # Replace the population with the children
        population[:population_size - 1, :, :] = children

    return best_fitness_values, avg_fitness_values


def main():
    pygame.init()
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    population = generate_population(START_POSITION, END_POSITION)

    for i in range(MAX_GENERATIONS):
        population = evolve_population(population, i)
        surface.fill((255, 255, 255))

        draw_obstacles(surface)  # draws rectangle for obstacles
        draw_start_and_end(surface)  # circle for start and end
        label_start_end(surface)  # label for start and end
        label_obstacles(surface)

        best_path = sorted(population, key=fitness, reverse=True)[0]
        pn = []
        for j in range(len(best_path) - 2):
            delta = ((dd(best_path[j + 1], best_path[j])) * alpha) + ((dd(best_path[j + 2], best_path[j])) * -alpha)
            # pn_ = tuple(best_path[i][0] + delta)+tuple(best_path[i][1]+delta)

            x, y = best_path[j]
            pn_ = (round(x + delta), round(y + delta))
            pn.append(pn_)
        pn.append(best_path[-1])
        print("path", pn)

        final_feasible_path(pn, i)
        draw_path_points(pn, surface)  # drawing node in path...... dots for node
        draw_path(pn, surface)  # draws line for path
        label_path_distance(surface, pn)  # prints length of distance of path
        find_mutate(pn, surface, i)  # finds all the points where mutation takes place

        pygame.display.update()
        #pygame.time.wait(500)

    min_distance = min(overall_best_path_dist)
    index = overall_best_path_dist.index(min_distance)
    print('\nThe most feasible path has a distance :', min_distance, 'units')
    print('\nThe path belongs to ', index + 1, 'generation\n')
    #print(infes_path)
    #print(len(infes_path))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()


best_fitness_values, avg_fitness_values = genetic_algorithm(5, 10, 0.2, 0.1)
plt.plot(best_fitness_values, label='Best Fitness')
plt.plot(avg_fitness_values, label='Average Fitness')
plt.legend()
plt.title('Fitness vs Avg Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.show()

if __name__ == '__main__':
    main()

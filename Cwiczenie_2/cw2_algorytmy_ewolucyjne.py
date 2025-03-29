# Cel zadania polega na implementacji algorytmu genetycznego z mutacją,
# selekcją ruletkową, krzyżowaniem jednopunktowym oraz sukcesją generacyjną.
import random
import numpy as np
import matplotlib.pyplot as plt
import time


cities = {
    "A": (0, 0), "B": (1, 3), "C": (2, 1), "D": (4, 6), "E": (5, 2),
    "F": (6, 5), "G": (8, 7), "H": (9, 4), "I": (10, 8), "J": (12, 3)
}

num_cities = len(cities)
distance_matrix = {}


def calculate_distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def create_distance_matrix():
    distance_matrix = {}
    for city1 in cities:
        distance_matrix[city1] = {}
        for city2 in cities:
            if city1 != city2:
                distance_matrix[city1][city2] = calculate_distance(city1, city2)
    return distance_matrix


def total_distance(route):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    total += distance_matrix[route[-1]][route[0]]
    return total


def fitness(route):
    return 1 / total_distance(route)


def initialize_population(size):
    population = []
    cities_list = list(cities.keys())
    for _ in range(size):
        route = cities_list.copy()
        random.shuffle(route)
        population.append(route)
    return population


def roulette_wheel_selection(population):
    fitness_values = [fitness(route) for route in population]
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)

    if max_fitness == min_fitness:
        return random.choice(population)

    scaled_fitness = [(f - min_fitness) / (max_fitness - min_fitness)
                      for f in fitness_values]

    # Normalize the probabilities
    total = sum(scaled_fitness)
    probabilities = [f / total for f in scaled_fitness] if total > 0 else [1 / len(population)] * len(population)

    return population[np.random.choice(len(population), p=probabilities)]


def single_point_crossover(parent1, parent2):
    point = random.randint(1, num_cities - 1)
    child = parent1[:point]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child


def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


def genetic_algorithm(population_size=100, generations=500, mutation_rate=0.1):
    population = initialize_population(population_size)
    best_route = None
    still_generations = 0

    for _ in range(generations):
        new_population = []

        for _ in range(population_size):
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)

            the_same_count = 0
            while parent1 == parent2:
                parent2 = roulette_wheel_selection(population)
                the_same_count += 1
                if the_same_count > 10:
                    parent2 = random.choice(population)
                    break

            child = single_point_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        current_best = min(population, key=total_distance)
        if (best_route is None or
           total_distance(current_best) < total_distance(best_route)):
            best_route = current_best
            still_generations = 0
        else:
            still_generations += 1

        if still_generations >= 10:
            break

    return best_route, total_distance(best_route)


def plot_population_size(population_sizes, generations):
    distances = []
    distances_average = []
    for population_size in population_sizes:
        for _ in range(10):
            _, best_distance = genetic_algorithm(population_size, generations)
            distances.append(best_distance)
        distances_average.append(np.mean(distances))
        distances.clear()

    plt.plot(population_sizes, distances_average, marker='o')
    plt.xlabel("Population size")
    plt.ylabel("Average distance")
    plt.title("Average distance vs Population size")
    plt.show()


def plot_generations(population_size, generations):
    distances = []
    distances_average = []
    for generation in generations:
        for _ in range(10):
            _, best_distance = genetic_algorithm(population_size, generation)
            distances.append(best_distance)
        distances_average.append(np.mean(distances))
        distances.clear()

    plt.plot(generations, distances_average, marker='o')
    plt.xlabel("Population size")
    plt.ylabel("Average distance")
    plt.title("Average distance vs Generations")
    plt.show()


def visualize_route(route):
    x = [cities[city][0] for city in route] + [cities[route[0]][0]]
    y = [cities[city][1] for city in route] + [cities[route[0]][1]]

    plt.plot(x, y, marker='o')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], f" {city}", fontsize=12, ha='right')

    plt.title("Route Visualization")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.show()

if __name__ == "__main__":
    distance_matrix = create_distance_matrix()
    # plot_population_size([10, 50, 100, 200, 500, 1000], 10)
    # plot_generations(100, [10, 50, 100, 200, 500, 1000])

    best_route, best_distance = genetic_algorithm(500, 50, 0.1)
    print("Best route:", best_route)
    print("Best distance:", best_distance)
    visualize_route(best_route)

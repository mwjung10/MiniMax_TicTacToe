#Cel zadania polega na implementacji algorytmu genetycznego z mutacją, selekcją ruletkową, krzyżowaniem jednopunktowym oraz sukcesją generacyjną.
import random
import numpy as np

cities = {
    "A": (0, 0), "B": (1, 3), "C": (2, 1), "D": (4, 6), "E": (5, 2),
    "F": (6, 5), "G": (8, 7), "H": (9, 4), "I": (10, 8), "J": (12, 3)
}

num_cities = len(cities)


def calculate_distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def total_distance(route):
    total = 0
    for i in range(len(route) - 1):
        total += calculate_distance(route[i], route[i + 1])
    return total + calculate_distance(route[-1], route[0])


def fitness(route):
    return 1 / total_distance(route)


def initalize_population(size):
    population = []
    cities_list = list(cities.keys())
    for _ in range(size):
        route = cities_list.copy()
        random.shuffle(route)
        population.append(route)
    return population


def roulette_wheel_selection(population):
    fitness_values = [total_distance(route) for route in population]
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    scaled_fitness = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness_values]
    probablities = [f / sum(scaled_fitness) for f in scaled_fitness]

    # Normalize the probabilities
    total = sum(probablities)
    probablities = [p / total for p in probablities]

    select_index = np.random.choice(len(population), p=probablities)
    return population[select_index]


def single_point_crossover(parent1, parent2):
    point = random.randint(1, num_cities - 1)
    child = parent1[:point]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child


def mutate(route, mutation_rate=0.1):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(route) - 1)
            route[i], route[swap_index] = route[swap_index], route[i]
    return route


def genetic_algorithm(population_size=100, generations=500, mutation_rate=0.1, elitism=True):
    population = initalize_population(population_size)
    best_route = None

    for _ in range(generations):
        new_population = []

        if elitism:
            best_route = min(population, key=lambda r: total_distance(r))
            new_population.append(best_route)

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)

            while parent1 == parent2:
                parent2 = roulette_wheel_selection(population)

            child = single_point_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        current_best = min(population, key=lambda r: total_distance(r))
        if best_route is None or total_distance(current_best) < total_distance(best_route):
            best_route = current_best

    return best_route, total_distance(best_route)


best_route, best_distance = genetic_algorithm(population_size=100, generations=500, mutation_rate=0.1, elitism=True)
print("Best route:", best_route)
print("Best distance:", best_distance)
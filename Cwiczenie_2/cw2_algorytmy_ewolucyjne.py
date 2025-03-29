# Cel zadania polega na implementacji algorytmu genetycznego z mutacją,
# selekcją ruletkową, krzyżowaniem jednopunktowym oraz sukcesją generacyjną.
import random
import numpy as np
import matplotlib.pyplot as plt


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
                distance_matrix[city1][city2] = calculate_distance(city1,
                                                                   city2)
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
    probabilities = ([f / total for f in scaled_fitness] if total > 0
                     else [1 / len(population)] * len(population))

    return population[np.random.choice(len(population), p=probabilities)]


def tournament_selection(population, tournament_size=100):
    tournament = random.sample(population, tournament_size)
    best_route = min(tournament, key=total_distance)
    return best_route


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


def genetic_algorithm(population_size=100, generations=500, mutation_rate=0.1,
                      selectionFunction=roulette_wheel_selection):
    population = initialize_population(population_size)
    best_route = None
    still_generations = 0

    for _ in range(generations):
        new_population = []

        for _ in range(population_size):
            parent1 = selectionFunction(population)
            parent2 = selectionFunction(population)

            the_same_count = 0
            while parent1 == parent2:
                parent2 = selectionFunction(population)
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
    for i, distance in enumerate(distances_average):
        plt.text(population_sizes[i], distance, f"{distance:.2f}", fontsize=9,
                 ha='right')
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
    for i, distance in enumerate(distances_average):
        plt.text(generations[i], distance, f"{distance:.2f}", fontsize=9,
                 ha='right')
    plt.xlabel("Generation size")
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


def test_mutation_rates(best_population, best_generations, mutation_rates):
    # we check every 20% of generations
    generations_list = list(range(1, best_generations + 1,
                                  best_generations // 5))
    results = {rate: [] for rate in mutation_rates}

    for mutation_rate in mutation_rates:
        for generations in generations_list:
            distances = []
            for _ in range(5):
                _, best_distance = genetic_algorithm(best_population,
                                                     generations,
                                                     mutation_rate)
                distances.append(best_distance)

            avg_distance = np.mean(distances)
            results[mutation_rate].append((generations, avg_distance))
            print(f"Mutation rate: {mutation_rate}, Generations: {generations}\
                  => Average distance: {avg_distance}")

    return results


def plot_mutation_results(mutation_results):
    for rate, data in mutation_results.items():
        generations, distances = zip(*data)
        plt.plot(generations, distances, marker='o', label=f"Mutacja: {rate}")

    plt.xlabel("Generations")
    plt.ylabel("Average distance")
    plt.title("Average distance vs Generations for different mutation rates")
    plt.legend()
    plt.show()


def compare_selection_methods(population_size, generations):
    methods = {
        "Roulette Wheel": roulette_wheel_selection,
        "Tournament": tournament_selection
    }
    results = {}

    for method_name, method in methods.items():
        distances = []
        for _ in range(10):
            _, best_distance = genetic_algorithm(population_size, generations,
                                                 selectionFunction=method)
            distances.append(best_distance)
        results[method_name] = np.mean(distances)

    return results


def plot_selection_methods(results):
    methods = list(results.keys())
    distances = list(results.values())

    plt.bar(methods, distances, color=['blue', 'orange'])
    plt.text(methods[0], distances[0], f"{distances[0]:.2f}",   fontsize=12,
             ha='center', va='bottom')
    plt.text(methods[1], distances[1], f"{distances[1]:.2f}", fontsize=12,
             ha='center', va='bottom')
    plt.ylabel("Average distance")
    plt.title("Average distance for different selection methods for population size 100")
    plt.show()


if __name__ == "__main__":
    distance_matrix = create_distance_matrix()
    # plot_population_size([10, 50, 100, 200, 500, 1000], 10)
    # plot_generations(100, [10, 50, 100, 200, 500, 1000])

    best_route, best_distance = genetic_algorithm(500, 50, 0.1)
    print("Best route:", best_route)
    print("Best distance:", best_distance)
    visualize_route(best_route)

    # plot_mutation_results(
    #     test_mutation_rates(500, 50, [0.05, 0.1, 0.2, 0.5, 1.0])
    # )

    # plot_selection_methods(
    #     compare_selection_methods(100, 50)
    # )

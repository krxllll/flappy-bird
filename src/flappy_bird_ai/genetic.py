import numpy as np


POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.15
INITIAL_MUTATION_STRENGTH = 0.2
MIN_MUTATION_STRENGTH = 0.05
MUTATION_STRENGTH = INITIAL_MUTATION_STRENGTH
ELITE_SIZE = 5
TARGET_PIPES = 10
EARLY_STOP_ON_TARGET = True
VALIDATION_SEEDS = list(range(10000, 10020))
TEST_SEEDS = list(range(20000, 20050))
RANDOM_IMMIGRANT_RATIO = 0.1
CHAMPION_OFFSPRING_RATIO = 0.15
CHAMPION_MUTATION_RATE = 0.03
CHAMPION_MUTATION_STRENGTH = 0.02


def initialize_population(pop_size, chromosome_length):
    """Create a starting population of random chromosomes."""
    return np.random.uniform(-1.0, 1.0, (pop_size, chromosome_length))


def crossover(parent1, parent2):
    """Combine two parent chromosomes using uniform crossover."""
    mask = np.random.rand(len(parent1)) > 0.5
    return np.where(mask, parent1, parent2)


def mutate(chromosome, mutation_strength=MUTATION_STRENGTH, mutation_rate=MUTATION_RATE):
    """Introduce random weight variations into a chromosome."""
    mutation_mask = np.random.rand(len(chromosome)) < mutation_rate
    noise = np.random.normal(0, mutation_strength, size=len(chromosome))
    chromosome[mutation_mask] += noise[mutation_mask]
    return chromosome


def adaptive_mutation_strength(
    generation_index,
    total_generations=GENERATIONS,
    initial_strength=INITIAL_MUTATION_STRENGTH,
    min_strength=MIN_MUTATION_STRENGTH,
):
    """Reduce mutation strength over time while preserving a minimum."""
    progress = generation_index / max(1, total_generations)
    return max(min_strength, initial_strength * (1 - progress))


def sort_population_indices(evaluation_results):
    """Rank candidates by pipes first, then fitness, then survival frames."""
    return sorted(
        range(len(evaluation_results)),
        key=lambda i: (
            evaluation_results[i]["pipes_passed"],
            evaluation_results[i]["fitness"],
            evaluation_results[i]["frames"],
        ),
        reverse=True,
    )


def create_next_generation(
    population,
    chromosome_length=None,
    pop_size=POPULATION_SIZE,
    elite_size=ELITE_SIZE,
    mutation_strength=MUTATION_STRENGTH,
    mutation_rate=MUTATION_RATE,
    global_best_genome=None,
    champion_offspring_ratio=CHAMPION_OFFSPRING_RATIO,
    random_immigrant_ratio=RANDOM_IMMIGRANT_RATIO,
    champion_mutation_rate=CHAMPION_MUTATION_RATE,
    champion_mutation_strength=CHAMPION_MUTATION_STRENGTH,
):
    """Create a new generation with elitism, champion injection, and immigrants."""
    next_generation = []

    if chromosome_length is None:
        chromosome_length = population.shape[1]

    if global_best_genome is not None:
        next_generation.append(global_best_genome.copy())

    for elite_index in range(elite_size):
        if len(next_generation) >= pop_size:
            break
        next_generation.append(population[elite_index].copy())

    if global_best_genome is not None:
        champion_offspring_count = int(pop_size * champion_offspring_ratio)
        for _ in range(champion_offspring_count):
            if len(next_generation) >= pop_size:
                break
            champion_child = global_best_genome.copy()
            next_generation.append(
                mutate(
                    champion_child,
                    mutation_strength=champion_mutation_strength,
                    mutation_rate=champion_mutation_rate,
                )
            )

    random_immigrant_count = int(pop_size * random_immigrant_ratio)
    parent_pool_size = min(len(population), max(1, int(pop_size * 0.4)))

    while len(next_generation) < pop_size - random_immigrant_count:
        p1_idx = np.random.randint(0, parent_pool_size)
        p2_idx = np.random.randint(0, parent_pool_size)
        child = crossover(population[p1_idx], population[p2_idx]).copy()
        next_generation.append(
            mutate(
                child,
                mutation_strength=mutation_strength,
                mutation_rate=mutation_rate,
            )
        )

    while len(next_generation) < pop_size:
        next_generation.append(np.random.uniform(-1.0, 1.0, chromosome_length))

    return np.array(next_generation)


def _result_key(result):
    pipes = result.get(
        "validated_mean_pipes",
        result.get("mean_pipes", result.get("pipes_passed", 0)),
    )
    fitness = result.get(
        "validated_mean_fitness",
        result.get("mean_fitness", result.get("fitness", 0.0)),
    )
    frames = result.get("mean_frames", result.get("frames", 0.0))
    return pipes, fitness, frames


def is_better_result(candidate, current_best):
    """Compare candidates using the baseline pipes, fitness, frames ordering."""
    if current_best is None:
        return True

    return _result_key(candidate) > _result_key(current_best)


def target_reached(evaluation_result, target_pipes=TARGET_PIPES):
    """Return True when an evaluation result meets the pipe target."""
    pipes = evaluation_result.get(
        "validated_mean_pipes",
        evaluation_result.get("mean_pipes", evaluation_result.get("pipes_passed", 0)),
    )
    return pipes >= target_pipes

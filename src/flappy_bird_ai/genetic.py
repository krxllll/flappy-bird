import numpy as np


POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.15
INITIAL_MUTATION_STRENGTH = 0.2
MIN_MUTATION_STRENGTH = 0.05
MUTATION_STRENGTH = INITIAL_MUTATION_STRENGTH
ELITE_SIZE = 5
TARGET_PIPES = 10
EARLY_STOP_ON_TARGET = True
TOP_VALIDATION_COUNT = 5
VALIDATION_EPISODES = 3


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


def create_next_generation(
    population,
    pop_size=POPULATION_SIZE,
    elite_size=ELITE_SIZE,
    mutation_strength=MUTATION_STRENGTH,
):
    """Create a new generation while preserving copied elites unchanged."""
    next_generation = []

    for elite_index in range(elite_size):
        next_generation.append(population[elite_index].copy())

    parent_pool_size = max(1, int(pop_size * 0.4))

    while len(next_generation) < pop_size:
        p1_idx = np.random.randint(0, parent_pool_size)
        p2_idx = np.random.randint(0, parent_pool_size)

        child = crossover(population[p1_idx], population[p2_idx]).copy()
        child = mutate(child, mutation_strength=mutation_strength)
        next_generation.append(child)

    return np.array(next_generation)


def is_better_result(candidate, current_best):
    """Compare results by pipes first, then fitness."""
    if current_best is None:
        return True

    candidate_key = (candidate["pipes_passed"], candidate["fitness"])
    current_key = (current_best["pipes_passed"], current_best["fitness"])
    return candidate_key > current_key


def target_reached(evaluation_result, target_pipes=TARGET_PIPES):
    """Return True when an evaluation result meets the pipe target."""
    return evaluation_result["pipes_passed"] >= target_pipes

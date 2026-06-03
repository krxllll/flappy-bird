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
TOP_VALIDATION_COUNT = 10
GENERATION_BEST_REEVALUATION_COUNT = 10
GENERATION_BEST_REEVALUATION_EPISODES = 3
VALIDATION_EPISODES = 20
CHAMPION_EVALUATION_EPISODES = 20
CHAMPION_OFFSPRING_RATIO = 0.35
CHAMPION_MUTATION_RATE = 0.03
CHAMPION_MUTATION_STRENGTH = 0.02
RANDOM_IMMIGRANT_RATIO = 0.1
BREAKTHROUGH_PIPES = 10
POST_BREAKTHROUGH_MUTATION_RATE = 0.05
POST_BREAKTHROUGH_MUTATION_STRENGTH = 0.03
PATIENCE_GENERATIONS = 15
PATIENCE_RANDOM_IMMIGRANT_RATIO = 0.25
PATIENCE_MUTATION_RATE = 0.20
PATIENCE_MUTATION_STRENGTH_MULTIPLIER = 1.25
PATIENCE_WEAK_CHAMPION_PIPES = 5
PATIENCE_WEAK_CHAMPION_OFFSPRING_RATIO = 0.10


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


def adapt_mutation_after_breakthrough(
    mutation_rate,
    mutation_strength,
    global_best_pipes,
    breakthrough_pipes=BREAKTHROUGH_PIPES,
):
    """Tighten mutation after the champion reaches a pipe breakthrough."""
    if global_best_pipes < breakthrough_pipes:
        return mutation_rate, mutation_strength

    return (
        min(mutation_rate, POST_BREAKTHROUGH_MUTATION_RATE),
        min(mutation_strength, POST_BREAKTHROUGH_MUTATION_STRENGTH),
    )


def calculate_champion_score(
    validated_mean_pipes,
    validated_min_pipes,
    validated_std_pipes,
):
    """Score champion consistency as a secondary tie-breaker."""
    return float(
        validated_mean_pipes
        + 0.3 * validated_min_pipes
        - 0.1 * validated_std_pipes
    )


def apply_patience_diversity(
    mutation_rate,
    mutation_strength,
    random_immigrant_ratio,
    champion_offspring_ratio,
    generations_without_improvement,
    validated_mean_pipes=0.0,
    patience_generations=PATIENCE_GENERATIONS,
):
    """Increase diversity after a long champion-improvement plateau."""
    if generations_without_improvement < patience_generations:
        return mutation_rate, mutation_strength, random_immigrant_ratio, champion_offspring_ratio

    mutation_rate = max(mutation_rate, PATIENCE_MUTATION_RATE)
    mutation_strength = mutation_strength * PATIENCE_MUTATION_STRENGTH_MULTIPLIER
    random_immigrant_ratio = max(random_immigrant_ratio, PATIENCE_RANDOM_IMMIGRANT_RATIO)

    if validated_mean_pipes < PATIENCE_WEAK_CHAMPION_PIPES:
        champion_offspring_ratio = min(
            champion_offspring_ratio,
            PATIENCE_WEAK_CHAMPION_OFFSPRING_RATIO,
        )

    return (
        mutation_rate,
        mutation_strength,
        random_immigrant_ratio,
        champion_offspring_ratio,
    )


def get_champion_offspring_ratio(validated_mean_pipes):
    """Use less champion cloning until the champion is actually strong."""
    if validated_mean_pipes >= 10:
        return 0.35
    if validated_mean_pipes >= 5:
        return 0.25
    return 0.15


def sort_population_indices(evaluation_results):
    """Rank candidates by pipes first, then fitness, then frames."""
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
    """Create a new generation while preserving elites and the champion."""
    next_generation = []

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
            champion_child = mutate(
                champion_child,
                mutation_strength=champion_mutation_strength,
                mutation_rate=champion_mutation_rate,
            )
            next_generation.append(champion_child)

    if chromosome_length is None:
        chromosome_length = population.shape[1]

    random_immigrant_count = int(pop_size * random_immigrant_ratio)
    reserved_for_random = random_immigrant_count
    parent_pool_size = min(len(population), max(1, int(pop_size * 0.4)))

    while len(next_generation) < pop_size - reserved_for_random:
        p1_idx = np.random.randint(0, parent_pool_size)
        p2_idx = np.random.randint(0, parent_pool_size)

        child = crossover(population[p1_idx], population[p2_idx]).copy()
        child = mutate(
            child,
            mutation_strength=mutation_strength,
            mutation_rate=mutation_rate,
        )
        next_generation.append(child)

    while len(next_generation) < pop_size:
        next_generation.append(
            np.random.uniform(-1.0, 1.0, chromosome_length)
        )

    return np.array(next_generation)


def is_better_result(candidate, current_best):
    """Compare validated champions by mean pipes, stability, then fitness."""
    if current_best is None:
        return True

    candidate_key = (
        candidate["validated_mean_pipes"],
        candidate["champion_score"],
        candidate["validated_mean_fitness"],
    )
    current_key = (
        current_best["validated_mean_pipes"],
        current_best["champion_score"],
        current_best["validated_mean_fitness"],
    )
    return candidate_key > current_key


def target_reached(evaluation_result, target_pipes=TARGET_PIPES):
    """Return True when an evaluation result meets the pipe target."""
    pipes = evaluation_result.get(
        "validated_mean_pipes",
        evaluation_result.get("pipes_passed", 0),
    )
    return pipes >= target_pipes

import numpy as np


POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.2
ELITE_SIZE = 5
TARGET_PIPES = 10
EARLY_STOP_ON_TARGET = True


def initialize_population(pop_size, chromosome_length):
    """Create a starting population of random chromosomes."""
    return np.random.uniform(-1.0, 1.0, (pop_size, chromosome_length))


def crossover(parent1, parent2):
    """Combine two parent chromosomes using uniform crossover."""
    mask = np.random.rand(len(parent1)) > 0.5
    return np.where(mask, parent1, parent2)


def mutate(chromosome):
    """Introduce random weight variations into a chromosome."""
    mutation_mask = np.random.rand(len(chromosome)) < MUTATION_RATE
    noise = np.random.normal(0, MUTATION_STRENGTH, size=len(chromosome))
    chromosome[mutation_mask] += noise[mutation_mask]
    return chromosome


def target_reached(evaluation_result, target_pipes=TARGET_PIPES):
    """Return True when an evaluation result meets the pipe target."""
    return evaluation_result["pipes_passed"] >= target_pipes

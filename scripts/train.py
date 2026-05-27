import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent
from flappy_bird_ai.genetic import (  # noqa: E402
    ELITE_SIZE,
    EARLY_STOP_ON_TARGET,
    GENERATIONS,
    POPULATION_SIZE,
    TARGET_PIPES,
    crossover,
    initialize_population,
    mutate,
    target_reached,
)
from flappy_bird_ai.simulation import evaluate_chromosome  # noqa: E402


BEST_GENOME_PATH = PROJECT_ROOT / "outputs" / "best_bird_genome.npy"
FITNESS_CHART_PATH = PROJECT_ROOT / "outputs" / "fitness_progression.png"
PIPE_CHART_PATH = PROJECT_ROOT / "outputs" / "pipe_progression.png"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "outputs" / "training_history.csv"


def save_training_history(history, path):
    fieldnames = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "best_pipes",
        "mean_pipes",
        "best_frames",
        "mean_frames",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_progress_plots(history):
    generations = [row["generation"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(
        generations,
        [row["best_fitness"] for row in history],
        label="Best Fitness",
        color="green",
        linewidth=2,
    )
    plt.plot(
        generations,
        [row["mean_fitness"] for row in history],
        label="Mean Fitness",
        color="orange",
        linestyle="--",
    )
    plt.title("Flappy Bird Genetic Algorithm Fitness Progression")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FITNESS_CHART_PATH)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        generations,
        [row["best_pipes"] for row in history],
        label="Best Pipes",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        generations,
        [row["mean_pipes"] for row in history],
        label="Mean Pipes",
        color="purple",
        linestyle="--",
    )
    plt.title("Flappy Bird Pipe Progression")
    plt.xlabel("Generation")
    plt.ylabel("Pipes Passed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PIPE_CHART_PATH)
    plt.close()


def train_evolutionary_ai():
    agent_meta = BirdAgent()
    chrom_len = agent_meta.chromosome_length

    BEST_GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)

    population = initialize_population(POPULATION_SIZE, chrom_len)

    training_history = []
    best_fitness_so_far = float("-inf")

    print(
        "Starting evolutionary training. "
        f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | "
        f"Target pipes: {TARGET_PIPES}\n"
    )

    for g in range(GENERATIONS):
        evaluation_results = []

        for i in range(POPULATION_SIZE):
            result = evaluate_chromosome(population[i], render=False)
            evaluation_results.append(result)

        fitness_scores = np.array([result["fitness"] for result in evaluation_results])
        pipes_scores = np.array([result["pipes_passed"] for result in evaluation_results])
        frame_scores = np.array([result["frames"] for result in evaluation_results])

        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]
        pipes_scores = pipes_scores[sorted_indices]
        frame_scores = frame_scores[sorted_indices]
        evaluation_results = [evaluation_results[idx] for idx in sorted_indices]

        best_fit = fitness_scores[0]
        mean_fit = np.mean(fitness_scores)
        best_pipes = int(pipes_scores[0])
        mean_pipes = np.mean(pipes_scores)
        best_frames = int(frame_scores[0])
        mean_frames = np.mean(frame_scores)

        training_history.append(
            {
                "generation": g + 1,
                "best_fitness": best_fit,
                "mean_fitness": mean_fit,
                "best_pipes": best_pipes,
                "mean_pipes": mean_pipes,
                "best_frames": best_frames,
                "mean_frames": mean_frames,
            }
        )

        print(
            f"Generation {g + 1:02d} | Best fitness: {best_fit:.1f} | "
            f"Avg fitness: {mean_fit:.1f} | Best pipes: {best_pipes} | "
            f"Avg pipes: {mean_pipes:.1f} | Best frames: {best_frames}"
        )

        if best_fit > best_fitness_so_far:
            best_fitness_so_far = best_fit
            np.save(BEST_GENOME_PATH, population[0])

        if target_reached(evaluation_results[0], TARGET_PIPES):
            np.save(BEST_GENOME_PATH, population[0])
            print(f"Target reached: best bird passed {best_pipes} pipes")

            if EARLY_STOP_ON_TARGET:
                break

        next_generation = []

        for e in range(ELITE_SIZE):
            next_generation.append(population[e])

        while len(next_generation) < POPULATION_SIZE:
            parent_pool_idx = int(POPULATION_SIZE * 0.4)
            p1_idx = np.random.randint(0, parent_pool_idx)
            p2_idx = np.random.randint(0, parent_pool_idx)

            child = crossover(population[p1_idx], population[p2_idx])
            child = mutate(child)
            next_generation.append(child)

        population = np.array(next_generation)

    save_training_history(training_history, TRAINING_HISTORY_PATH)
    save_progress_plots(training_history)

    print(f"\nTraining complete! Best genome saved as '{BEST_GENOME_PATH}'")
    print(f"Training history saved as '{TRAINING_HISTORY_PATH}'")
    print(f"Fitness plot saved as '{FITNESS_CHART_PATH}'")
    print(f"Pipe plot saved as '{PIPE_CHART_PATH}'")


if __name__ == "__main__":
    train_evolutionary_ai()

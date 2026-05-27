import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent
from flappy_bird_ai.genetic import (  # noqa: E402
    CHAMPION_EVALUATION_EPISODES,
    EARLY_STOP_ON_TARGET,
    GENERATIONS,
    MUTATION_RATE,
    POPULATION_SIZE,
    TARGET_PIPES,
    TOP_VALIDATION_COUNT,
    VALIDATION_EPISODES,
    adapt_mutation_after_breakthrough,
    adaptive_mutation_strength,
    create_next_generation,
    initialize_population,
    is_better_result,
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
        "global_best_pipes",
        "global_best_fitness",
        "champion_mean_pipes",
        "champion_max_pipes",
        "champion_min_pipes",
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
        label="Generation Best Fitness",
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
    plt.plot(
        generations,
        [row["global_best_fitness"] for row in history],
        label="Validated Global Best Fitness",
        color="blue",
        linewidth=2,
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
        label="Generation Best Pipes",
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
    plt.plot(
        generations,
        [row["global_best_pipes"] for row in history],
        label="Validated Global Best Pipes",
        color="green",
        linewidth=2,
    )
    plt.plot(
        generations,
        [row["champion_mean_pipes"] for row in history],
        label="Champion Re-eval Mean Pipes",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    plt.title("Flappy Bird Pipe Progression")
    plt.xlabel("Generation")
    plt.ylabel("Pipes Passed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PIPE_CHART_PATH)
    plt.close()


def summarize_results(results):
    return {
        "fitness": float(np.mean([result["fitness"] for result in results])),
        "frames": float(np.mean([result["frames"] for result in results])),
        "pipes_passed": float(np.mean([result["pipes_passed"] for result in results])),
    }


def summarize_champion_pipes(results):
    pipes = [result["pipes_passed"] for result in results]
    return {
        "champion_mean_pipes": float(np.mean(pipes)),
        "champion_max_pipes": int(np.max(pipes)),
        "champion_min_pipes": int(np.min(pipes)),
    }


def validate_chromosome(chromosome, episodes=VALIDATION_EPISODES):
    results = [
        evaluate_chromosome(chromosome, render=False)
        for _ in range(episodes)
    ]
    return summarize_results(results)


def evaluate_champion(chromosome, episodes=CHAMPION_EVALUATION_EPISODES):
    results = [
        evaluate_chromosome(chromosome, render=False)
        for _ in range(episodes)
    ]
    return summarize_champion_pipes(results)


def train_evolutionary_ai():
    agent_meta = BirdAgent()
    chrom_len = agent_meta.chromosome_length

    BEST_GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)

    population = initialize_population(POPULATION_SIZE, chrom_len)

    training_history = []
    global_best_genome = None
    global_best_result = None

    print(
        "Starting evolutionary training. "
        f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | "
        f"Target pipes: {TARGET_PIPES} | Validation episodes: {VALIDATION_EPISODES}\n"
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

        validation_count = min(TOP_VALIDATION_COUNT, POPULATION_SIZE)
        for candidate_index in range(validation_count):
            validation_result = validate_chromosome(population[candidate_index])

            if is_better_result(validation_result, global_best_result):
                global_best_result = validation_result
                global_best_genome = population[candidate_index].copy()
                np.save(BEST_GENOME_PATH, global_best_genome)
                print(
                    "New global best saved | "
                    f"Validated pipes: {global_best_result['pipes_passed']:.2f} | "
                    f"Validated fitness: {global_best_result['fitness']:.1f}"
                )

        global_best_pipes = (
            global_best_result["pipes_passed"] if global_best_result is not None else 0.0
        )
        global_best_fitness = (
            global_best_result["fitness"] if global_best_result is not None else 0.0
        )
        champion_stats = {
            "champion_mean_pipes": 0.0,
            "champion_max_pipes": 0,
            "champion_min_pipes": 0,
        }

        if global_best_genome is not None:
            champion_stats = evaluate_champion(global_best_genome)

        training_history.append(
            {
                "generation": g + 1,
                "best_fitness": best_fit,
                "mean_fitness": mean_fit,
                "best_pipes": best_pipes,
                "mean_pipes": mean_pipes,
                "best_frames": best_frames,
                "mean_frames": mean_frames,
                "global_best_pipes": global_best_pipes,
                "global_best_fitness": global_best_fitness,
                **champion_stats,
            }
        )

        print(
            f"Generation {g + 1:02d} | Best fitness: {best_fit:.1f} | "
            f"Avg fitness: {mean_fit:.1f} | Best pipes: {best_pipes} | "
            f"Avg pipes: {mean_pipes:.1f} | Best frames: {best_frames} | "
            f"Global best pipes: {global_best_pipes:.2f} | "
            f"Global best fitness: {global_best_fitness:.1f} | "
            f"Champion pipes mean/max/min: "
            f"{champion_stats['champion_mean_pipes']:.2f}/"
            f"{champion_stats['champion_max_pipes']}/"
            f"{champion_stats['champion_min_pipes']}"
        )

        champion_target_result = {
            "pipes_passed": champion_stats["champion_mean_pipes"],
            "fitness": global_best_fitness,
        }

        if global_best_genome is not None and target_reached(champion_target_result, TARGET_PIPES):
            print(
                "Target confirmed by champion re-evaluation: "
                f"champion averaged {champion_stats['champion_mean_pipes']:.2f} pipes"
            )

            if EARLY_STOP_ON_TARGET:
                break

        current_mutation_strength = adaptive_mutation_strength(g, GENERATIONS)
        current_mutation_rate = MUTATION_RATE
        current_mutation_rate, current_mutation_strength = adapt_mutation_after_breakthrough(
            current_mutation_rate,
            current_mutation_strength,
            global_best_pipes,
        )
        population = create_next_generation(
            population,
            pop_size=POPULATION_SIZE,
            mutation_strength=current_mutation_strength,
            mutation_rate=current_mutation_rate,
            global_best_genome=global_best_genome,
        )

    save_training_history(training_history, TRAINING_HISTORY_PATH)
    save_progress_plots(training_history)

    print(f"\nTraining complete! Best genome saved as '{BEST_GENOME_PATH}'")
    print(f"Training history saved as '{TRAINING_HISTORY_PATH}'")
    print(f"Fitness plot saved as '{FITNESS_CHART_PATH}'")
    print(f"Pipe plot saved as '{PIPE_CHART_PATH}'")


if __name__ == "__main__":
    train_evolutionary_ai()

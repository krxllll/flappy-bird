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
    GENERATION_BEST_REEVALUATION_COUNT,
    GENERATION_BEST_REEVALUATION_EPISODES,
    MUTATION_RATE,
    POPULATION_SIZE,
    RANDOM_IMMIGRANT_RATIO,
    TARGET_PIPES,
    TOP_VALIDATION_COUNT,
    VALIDATION_EPISODES,
    adapt_mutation_after_breakthrough,
    adaptive_mutation_strength,
    apply_patience_diversity,
    calculate_champion_score,
    create_next_generation,
    get_champion_offspring_ratio,
    initialize_population,
    is_better_result,
    sort_population_indices,
    target_reached,
)
from flappy_bird_ai.simulation import PIPE_EXPONENT, evaluate_chromosome  # noqa: E402


BEST_GENOME_PATH = PROJECT_ROOT / "outputs" / "best_bird_genome.npy"
FITNESS_CHART_PATH = PROJECT_ROOT / "outputs" / "fitness_progression.png"
PIPE_CHART_PATH = PROJECT_ROOT / "outputs" / "pipe_progression.png"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "outputs" / "training_history.csv"


def save_training_history(history, path):
    fieldnames = [
        "generation",
        "raw_generation_best_fitness",
        "revalidated_generation_best_fitness",
        "best_fitness",
        "mean_fitness",
        "best_pipes",
        "mean_pipes",
        "best_frames",
        "mean_frames",
        "validated_mean_pipes",
        "validated_min_pipes",
        "validated_max_pipes",
        "validated_std_pipes",
        "validated_mean_fitness",
        "champion_score",
        "generations_without_champion_improvement",
        "mutation_rate",
        "mutation_strength",
        "champion_offspring_ratio",
        "random_immigrant_ratio",
        "pipe_exponent",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_progress_plots(history):
    generations = [row["generation"] for row in history]
    raw_best_fitness = [row["raw_generation_best_fitness"] for row in history]
    best_fitness = [row["best_fitness"] for row in history]
    mean_fitness = [row["mean_fitness"] for row in history]
    best_pipes = [row["best_pipes"] for row in history]
    mean_pipes = [row["mean_pipes"] for row in history]
    validated_fitness = [row["validated_mean_fitness"] for row in history]
    validated_pipes = [row["validated_mean_pipes"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(
        generations,
        raw_best_fitness,
        label="Raw Generation Best Fitness",
        color="gray",
        linestyle="-",
        linewidth=1,
        alpha=0.35,
    )
    plt.plot(
        generations,
        best_fitness,
        label="Revalidated Generation Best Fitness",
        color="green",
        linewidth=1.5,
        alpha=0.8,
    )
    plt.plot(
        generations,
        mean_fitness,
        label="Mean Fitness",
        color="orange",
        linestyle="--",
    )
    plt.plot(
        generations,
        validated_fitness,
        label="Validated Champion Fitness",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        generations,
        moving_average(best_fitness),
        label="Revalidated Generation Best Fitness MA(5)",
        color="darkgreen",
        linestyle=":",
    )
    plt.plot(
        generations,
        moving_average(mean_fitness),
        label="Mean Fitness MA(5)",
        color="darkorange",
        linestyle=":",
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
        best_pipes,
        label="Generation Best Pipes",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        generations,
        mean_pipes,
        label="Mean Pipes",
        color="purple",
        linestyle="--",
    )
    plt.plot(
        generations,
        validated_pipes,
        label="Validated Champion Mean Pipes",
        color="green",
        linewidth=2,
    )
    plt.plot(
        generations,
        [row["validated_min_pipes"] for row in history],
        label="Validated Champion Min Pipes",
        color="red",
        linestyle="--",
    )
    plt.plot(
        generations,
        moving_average(best_pipes),
        label="Generation Best Pipes MA(5)",
        color="navy",
        linestyle=":",
    )
    plt.plot(
        generations,
        moving_average(mean_pipes),
        label="Mean Pipes MA(5)",
        color="purple",
        linestyle=":",
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
    pipes = np.array([result["pipes_passed"] for result in results], dtype=float)
    fitness = np.array([result["fitness"] for result in results], dtype=float)
    mean_pipes = float(np.mean(pipes))
    min_pipes = int(np.min(pipes))
    std_pipes = float(np.std(pipes))

    return {
        "validated_mean_pipes": mean_pipes,
        "validated_min_pipes": min_pipes,
        "validated_max_pipes": int(np.max(pipes)),
        "validated_std_pipes": std_pipes,
        "validated_mean_fitness": float(np.mean(fitness)),
        "champion_score": calculate_champion_score(mean_pipes, min_pipes, std_pipes),
    }


def summarize_generation_results(results):
    pipes = np.array([result["pipes_passed"] for result in results], dtype=float)
    fitness = np.array([result["fitness"] for result in results], dtype=float)
    frames = np.array([result["frames"] for result in results], dtype=float)

    return {
        "fitness": float(np.mean(fitness)),
        "pipes_passed": float(np.mean(pipes)),
        "frames": float(np.mean(frames)),
    }


def moving_average(values, window=5):
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(float(np.mean(values[start : index + 1])))
    return averaged


def validate_chromosome(chromosome, episodes=VALIDATION_EPISODES):
    results = [
        evaluate_chromosome(chromosome, render=False)
        for _ in range(episodes)
    ]
    return summarize_results(results)


def reevaluate_generation_candidate(
    chromosome,
    episodes=GENERATION_BEST_REEVALUATION_EPISODES,
):
    results = [
        evaluate_chromosome(chromosome, render=False)
        for _ in range(episodes)
    ]
    return summarize_generation_results(results)


def evaluate_champion(chromosome, episodes=CHAMPION_EVALUATION_EPISODES):
    results = [
        evaluate_chromosome(chromosome, render=False)
        for _ in range(episodes)
    ]
    return summarize_results(results)


def should_validate_candidate(candidate_result, current_best):
    if current_best is None:
        return True

    return candidate_result["pipes_passed"] >= current_best["validated_mean_pipes"]


def train_evolutionary_ai():
    agent_meta = BirdAgent()
    chrom_len = agent_meta.chromosome_length

    BEST_GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)

    population = initialize_population(POPULATION_SIZE, chrom_len)

    training_history = []
    global_best_genome = None
    global_best_result = None
    generations_without_champion_improvement = 0

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

        sorted_indices = sort_population_indices(evaluation_results)
        population = population[sorted_indices]
        evaluation_results = [evaluation_results[idx] for idx in sorted_indices]
        fitness_scores = np.array([result["fitness"] for result in evaluation_results])
        pipes_scores = np.array([result["pipes_passed"] for result in evaluation_results])
        frame_scores = np.array([result["frames"] for result in evaluation_results])

        raw_best_fit = fitness_scores[0]
        mean_fit = np.mean(fitness_scores)
        raw_best_pipes = int(pipes_scores[0])
        mean_pipes = np.mean(pipes_scores)
        raw_best_frames = int(frame_scores[0])
        mean_frames = np.mean(frame_scores)

        revalidation_count = min(GENERATION_BEST_REEVALUATION_COUNT, POPULATION_SIZE)
        revalidated_generation_results = [
            reevaluate_generation_candidate(population[candidate_index])
            for candidate_index in range(revalidation_count)
        ]
        generation_best_result = max(
            revalidated_generation_results,
            key=lambda result: (result["pipes_passed"], result["fitness"]),
        )

        best_fit = generation_best_result["fitness"]
        best_pipes = generation_best_result["pipes_passed"]
        best_frames = generation_best_result["frames"]

        current_mutation_strength = adaptive_mutation_strength(g, GENERATIONS)
        current_mutation_rate = MUTATION_RATE
        champion_improved = False

        validation_count = min(TOP_VALIDATION_COUNT, POPULATION_SIZE)
        for candidate_index in range(validation_count):
            if not should_validate_candidate(evaluation_results[candidate_index], global_best_result):
                continue

            validation_result = validate_chromosome(population[candidate_index])

            if is_better_result(validation_result, global_best_result):
                global_best_result = validation_result
                global_best_genome = population[candidate_index].copy()
                champion_improved = True
                np.save(BEST_GENOME_PATH, global_best_genome)
                print(
                    "New global best saved | "
                    f"Validated mean pipes: {global_best_result['validated_mean_pipes']:.2f} | "
                    f"Validated fitness: {global_best_result['validated_mean_fitness']:.1f} | "
                    f"Champion score: {global_best_result['champion_score']:.2f}"
                )

        validated_stats = {
            "validated_mean_pipes": 0.0,
            "validated_min_pipes": 0,
            "validated_max_pipes": 0,
            "validated_std_pipes": 0.0,
            "validated_mean_fitness": 0.0,
            "champion_score": 0.0,
        }

        if global_best_genome is not None:
            champion_result = evaluate_champion(global_best_genome)

            if is_better_result(champion_result, global_best_result):
                global_best_result = champion_result
                global_best_genome = global_best_genome.copy()
                champion_improved = True
                np.save(BEST_GENOME_PATH, global_best_genome)

            validated_stats = champion_result

        if champion_improved:
            generations_without_champion_improvement = 0
        else:
            generations_without_champion_improvement += 1

        breakthrough_pipes = (
            global_best_result["validated_mean_pipes"]
            if global_best_result is not None
            else 0.0
        )
        current_mutation_rate, current_mutation_strength = adapt_mutation_after_breakthrough(
            current_mutation_rate,
            current_mutation_strength,
            breakthrough_pipes,
        )
        current_random_immigrant_ratio = RANDOM_IMMIGRANT_RATIO
        current_champion_offspring_ratio = get_champion_offspring_ratio(breakthrough_pipes)
        (
            current_mutation_rate,
            current_mutation_strength,
            current_random_immigrant_ratio,
            current_champion_offspring_ratio,
        ) = apply_patience_diversity(
            current_mutation_rate,
            current_mutation_strength,
            current_random_immigrant_ratio,
            current_champion_offspring_ratio,
            generations_without_champion_improvement,
            breakthrough_pipes,
        )

        training_history.append(
            {
                "generation": g + 1,
                "raw_generation_best_fitness": raw_best_fit,
                "revalidated_generation_best_fitness": best_fit,
                "best_fitness": best_fit,
                "mean_fitness": mean_fit,
                "best_pipes": best_pipes,
                "mean_pipes": mean_pipes,
                "best_frames": best_frames,
                "mean_frames": mean_frames,
                **validated_stats,
                "generations_without_champion_improvement": generations_without_champion_improvement,
                "mutation_rate": current_mutation_rate,
                "mutation_strength": current_mutation_strength,
                "champion_offspring_ratio": current_champion_offspring_ratio,
                "random_immigrant_ratio": current_random_immigrant_ratio,
                "pipe_exponent": PIPE_EXPONENT,
            }
        )

        print(
            f"Generation {g + 1:02d} | Raw best fitness: {raw_best_fit:.1f} | "
            f"Revalidated best fitness: {best_fit:.1f} | "
            f"Avg fitness: {mean_fit:.1f} | Revalidated best pipes: {best_pipes:.2f} | "
            f"Avg pipes: {mean_pipes:.1f} | Revalidated best frames: {best_frames:.1f} | "
            f"Validated champion pipes mean/min/max/std: "
            f"{validated_stats['validated_mean_pipes']:.2f}/"
            f"{validated_stats['validated_min_pipes']}/"
            f"{validated_stats['validated_max_pipes']}/"
            f"{validated_stats['validated_std_pipes']:.2f} | "
            f"Champion score: {validated_stats['champion_score']:.2f} | "
            f"Patience: {generations_without_champion_improvement} | "
            f"Mutation: {current_mutation_rate:.3f}/{current_mutation_strength:.3f} | "
            f"Champion offspring: {current_champion_offspring_ratio:.2f} | "
            f"Random ratio: {current_random_immigrant_ratio:.2f}"
        )

        if global_best_genome is not None and target_reached(validated_stats, TARGET_PIPES):
            print(
                "Target confirmed by champion re-evaluation: "
                f"champion averaged {validated_stats['validated_mean_pipes']:.2f} pipes"
            )

            if EARLY_STOP_ON_TARGET:
                break

        population = create_next_generation(
            population,
            chromosome_length=chrom_len,
            pop_size=POPULATION_SIZE,
            mutation_strength=current_mutation_strength,
            mutation_rate=current_mutation_rate,
            global_best_genome=global_best_genome,
            champion_offspring_ratio=current_champion_offspring_ratio,
            random_immigrant_ratio=current_random_immigrant_ratio,
        )

    save_training_history(training_history, TRAINING_HISTORY_PATH)
    save_progress_plots(training_history)

    print(f"\nTraining complete! Best genome saved as '{BEST_GENOME_PATH}'")
    print(f"Training history saved as '{TRAINING_HISTORY_PATH}'")
    print(f"Fitness plot saved as '{FITNESS_CHART_PATH}'")
    print(f"Pipe plot saved as '{PIPE_CHART_PATH}'")


if __name__ == "__main__":
    train_evolutionary_ai()

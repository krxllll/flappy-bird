import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent
from flappy_bird_ai.genetic import (  # noqa: E402
    EARLY_STOP_ON_TARGET,
    GENERATIONS,
    MUTATION_RATE,
    POPULATION_SIZE,
    RANDOM_IMMIGRANT_RATIO,
    TARGET_PIPES,
    TEST_SEEDS,
    adaptive_mutation_strength,
    create_next_generation,
    initialize_population,
    is_better_result,
    sort_population_indices,
    target_reached,
)
from flappy_bird_ai.simulation import (  # noqa: E402
    PIPE_EXPONENT,
    evaluate_chromosome_on_seeds,
    evaluate_population,
    suppress_gymnasium_observation_warnings,
)


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
        "global_best_fitness",
        "global_best_pipes",
        "mutation_rate",
        "mutation_strength",
        "random_immigrant_ratio",
        "pipe_exponent",
        "population_eval_seconds",
        "next_generation_seconds",
        "generation_seconds",
        "test_mean_pipes",
        "test_min_pipes",
        "test_max_pipes",
        "test_std_pipes",
        "test_mean_fitness",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def moving_average(values, window=5):
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(float(np.mean(values[start : index + 1])))
    return averaged


def save_progress_plots(history):
    if not history:
        return

    generations = [row["generation"] for row in history]
    best_fitness = [row["best_fitness"] for row in history]
    mean_fitness = [row["mean_fitness"] for row in history]
    global_best_fitness = [row["global_best_fitness"] for row in history]
    best_pipes = [row["best_pipes"] for row in history]
    mean_pipes = [row["mean_pipes"] for row in history]
    global_best_pipes = [row["global_best_pipes"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label="Generation Best Fitness", color="green")
    plt.plot(generations, mean_fitness, label="Mean Fitness", color="orange", linestyle="--")
    plt.plot(generations, global_best_fitness, label="Global Best Fitness", color="blue")
    plt.plot(
        generations,
        moving_average(best_fitness),
        label="Generation Best Fitness MA(5)",
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
    plt.plot(generations, best_pipes, label="Generation Best Pipes", color="blue")
    plt.plot(generations, mean_pipes, label="Mean Pipes", color="purple", linestyle="--")
    plt.plot(generations, global_best_pipes, label="Global Best Pipes", color="green")
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
    frames = np.array([result["frames"] for result in results], dtype=float)

    return {
        "fitness": float(np.mean(fitness)),
        "pipes_passed": float(np.mean(pipes)),
        "frames": float(np.mean(frames)),
        "mean_fitness": float(np.mean(fitness)),
        "mean_pipes": float(np.mean(pipes)),
        "min_pipes": int(np.min(pipes)),
        "max_pipes": int(np.max(pipes)),
        "std_pipes": float(np.std(pipes)),
        "mean_frames": float(np.mean(frames)),
        "validated_mean_pipes": float(np.mean(pipes)),
        "validated_mean_fitness": float(np.mean(fitness)),
    }


def summarize_population(results):
    fitness = np.array([result["fitness"] for result in results], dtype=float)
    pipes = np.array([result["pipes_passed"] for result in results], dtype=float)
    frames = np.array([result["frames"] for result in results], dtype=float)

    return {
        "mean_fitness": float(np.mean(fitness)),
        "mean_pipes": float(np.mean(pipes)),
        "mean_frames": float(np.mean(frames)),
    }


def benchmark_champion(chromosome, seeds=TEST_SEEDS, workers=1, executor=None):
    results = evaluate_chromosome_on_seeds(
        chromosome,
        seeds=seeds,
        workers=workers,
        executor=executor,
    )
    return summarize_results(results)


def default_worker_count():
    return max(1, (os.cpu_count() or 1) - 1)


def resolve_worker_count(requested_workers=None, no_parallel=False):
    if no_parallel:
        return 1
    if requested_workers is None:
        return default_worker_count()
    return max(1, requested_workers)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Flappy Bird GA agent.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for population evaluation.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multiprocessing and evaluate population sequentially.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations to train. Defaults to the configured value.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Continue training through all requested generations after the target is reached.",
    )
    parser.add_argument(
        "--target-pipes",
        type=int,
        default=None,
        help="Override the configured target pipe count.",
    )
    return parser.parse_args(argv)


def should_stop_training(
    evaluation_result,
    target_pipes=TARGET_PIPES,
    early_stop_on_target=EARLY_STOP_ON_TARGET,
):
    return early_stop_on_target and target_reached(evaluation_result, target_pipes)


def train_evolutionary_ai(
    workers=None,
    generations=None,
    target_pipes=None,
    early_stop_on_target=None,
):
    suppress_gymnasium_observation_warnings()

    if workers is None:
        workers = default_worker_count()
    if generations is None:
        generations = GENERATIONS
    if target_pipes is None:
        target_pipes = TARGET_PIPES
    if early_stop_on_target is None:
        early_stop_on_target = EARLY_STOP_ON_TARGET

    agent_meta = BirdAgent()
    chromosome_length = agent_meta.chromosome_length

    BEST_GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)

    population = initialize_population(POPULATION_SIZE, chromosome_length)
    training_history = []
    global_best_genome = None
    global_best_result = None
    total_start_time = time.perf_counter()

    print(
        "Starting evolutionary training. "
        f"Population: {POPULATION_SIZE} | Generations: {generations} | "
        f"Workers: {workers} | Target pipes: {target_pipes} | "
        f"Early stopping: {'enabled' if early_stop_on_target else 'disabled'}\n"
    )
    print(f"Population evaluation: {'parallel' if workers > 1 else 'sequential'}\n")

    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=suppress_gymnasium_observation_warnings,
        )

    try:
        for generation_index in range(generations):
            generation = generation_index + 1
            generation_start_time = time.perf_counter()

            population_eval_start_time = time.perf_counter()
            evaluation_results = evaluate_population(
                population,
                workers=workers,
                executor=executor,
            )
            population_eval_seconds = time.perf_counter() - population_eval_start_time

            sorted_indices = sort_population_indices(evaluation_results)
            population = population[sorted_indices]
            evaluation_results = [evaluation_results[index] for index in sorted_indices]
            best_result = evaluation_results[0]
            population_summary = summarize_population(evaluation_results)

            if is_better_result(best_result, global_best_result):
                global_best_result = best_result.copy()
                global_best_genome = population[0].copy()
                np.save(BEST_GENOME_PATH, global_best_genome)
                print(
                    "New global best saved | "
                    f"Pipes: {global_best_result['pipes_passed']} | "
                    f"Fitness: {global_best_result['fitness']:.1f} | "
                    f"Frames: {global_best_result['frames']}"
                )

            mutation_strength = adaptive_mutation_strength(generation_index, generations)
            mutation_rate = MUTATION_RATE
            random_immigrant_ratio = RANDOM_IMMIGRANT_RATIO
            next_generation_seconds = 0.0

            target_met = global_best_result is not None and target_reached(
                global_best_result,
                target_pipes,
            )

            if not (
                target_met
                and should_stop_training(
                    global_best_result,
                    target_pipes=target_pipes,
                    early_stop_on_target=early_stop_on_target,
                )
            ):
                next_generation_start_time = time.perf_counter()
                population = create_next_generation(
                    population,
                    chromosome_length=chromosome_length,
                    pop_size=POPULATION_SIZE,
                    mutation_strength=mutation_strength,
                    mutation_rate=mutation_rate,
                    global_best_genome=global_best_genome,
                    random_immigrant_ratio=random_immigrant_ratio,
                )
                next_generation_seconds = time.perf_counter() - next_generation_start_time

            generation_seconds = time.perf_counter() - generation_start_time
            training_history.append(
                {
                    "generation": generation,
                    "best_fitness": best_result["fitness"],
                    "mean_fitness": population_summary["mean_fitness"],
                    "best_pipes": best_result["pipes_passed"],
                    "mean_pipes": population_summary["mean_pipes"],
                    "best_frames": best_result["frames"],
                    "mean_frames": population_summary["mean_frames"],
                    "global_best_fitness": global_best_result["fitness"],
                    "global_best_pipes": global_best_result["pipes_passed"],
                    "mutation_rate": mutation_rate,
                    "mutation_strength": mutation_strength,
                    "random_immigrant_ratio": random_immigrant_ratio,
                    "pipe_exponent": PIPE_EXPONENT,
                    "population_eval_seconds": population_eval_seconds,
                    "next_generation_seconds": next_generation_seconds,
                    "generation_seconds": generation_seconds,
                    "test_mean_pipes": "",
                    "test_min_pipes": "",
                    "test_max_pipes": "",
                    "test_std_pipes": "",
                    "test_mean_fitness": "",
                }
            )

            print(
                f"Generation {generation:02d} | "
                f"Best fitness: {best_result['fitness']:.1f} | "
                f"Avg fitness: {population_summary['mean_fitness']:.1f} | "
                f"Best pipes: {best_result['pipes_passed']} | "
                f"Avg pipes: {population_summary['mean_pipes']:.2f} | "
                f"Global best pipes: {global_best_result['pipes_passed']} | "
                f"Mutation: {mutation_rate:.3f}/{mutation_strength:.3f} | "
                f"Random immigrants: {random_immigrant_ratio:.2f}"
            )
            print(
                f"Generation {generation:02d} timing | "
                f"population={population_eval_seconds:.1f}s | "
                f"next_generation={next_generation_seconds:.1f}s | "
                f"total={generation_seconds:.1f}s | "
                f"workers={workers}"
            )

            if target_met and should_stop_training(
                global_best_result,
                target_pipes=target_pipes,
                early_stop_on_target=early_stop_on_target,
            ):
                print(
                    "Target reached by global best: "
                    f"{global_best_result['pipes_passed']} pipes "
                    f"(target: {target_pipes})"
                )
                break

        if global_best_genome is not None:
            print("Evaluating saved champion on held-out test seeds...")
            test_result = benchmark_champion(
                global_best_genome,
                seeds=TEST_SEEDS,
                workers=workers,
                executor=executor,
            )
            if training_history:
                training_history[-1]["test_mean_pipes"] = test_result["mean_pipes"]
                training_history[-1]["test_min_pipes"] = test_result["min_pipes"]
                training_history[-1]["test_max_pipes"] = test_result["max_pipes"]
                training_history[-1]["test_std_pipes"] = test_result["std_pipes"]
                training_history[-1]["test_mean_fitness"] = test_result["mean_fitness"]
            print(
                "Test evaluation | "
                f"test_mean_pipes={test_result['mean_pipes']:.2f} | "
                f"test_min_pipes={test_result['min_pipes']} | "
                f"test_max_pipes={test_result['max_pipes']} | "
                f"test_std_pipes={test_result['std_pipes']:.2f} | "
                f"test_mean_fitness={test_result['mean_fitness']:.1f}"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    save_training_history(training_history, TRAINING_HISTORY_PATH)
    save_progress_plots(training_history)

    print(f"\nTraining complete! Best genome saved as '{BEST_GENOME_PATH}'")
    print(f"Training history saved as '{TRAINING_HISTORY_PATH}'")
    print(f"Fitness plot saved as '{FITNESS_CHART_PATH}'")
    print(f"Pipe plot saved as '{PIPE_CHART_PATH}'")
    print(f"Total training time: {time.perf_counter() - total_start_time:.1f}s")


def main():
    args = parse_args()
    workers = resolve_worker_count(args.workers, args.no_parallel)
    train_evolutionary_ai(
        workers=workers,
        generations=args.generations,
        target_pipes=args.target_pipes,
        early_stop_on_target=not args.no_early_stop,
    )


if __name__ == "__main__":
    main()

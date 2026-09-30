import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
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
    CHAMPION_OFFSPRING_RATIO,
    EARLY_STOP_ON_TARGET,
    GENERATIONS,
    INITIAL_MUTATION_STRENGTH,
    MUTATION_RATE,
    POPULATION_SIZE,
    RANDOM_IMMIGRANT_RATIO,
    TARGET_PIPES,
    TEST_SEEDS,
    TOP_VALIDATION_COUNT,
    VALIDATION_SEEDS,
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
    evaluate_population_on_seed_batch,
    suppress_gymnasium_observation_warnings,
)


BEST_GENOME_PATH = PROJECT_ROOT / "outputs" / "best_bird_genome.npy"
BEST_METADATA_PATH = PROJECT_ROOT / "outputs" / "best_bird_metadata.json"
FITNESS_CHART_PATH = PROJECT_ROOT / "outputs" / "fitness_progression.png"
PIPE_CHART_PATH = PROJECT_ROOT / "outputs" / "pipe_progression.png"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "outputs" / "training_history.csv"

CSV_FIELDNAMES = [
    "generation",
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
    "mutation_rate",
    "mutation_strength",
    "random_immigrant_ratio",
    "champion_offspring_ratio",
    "generation_time",
    "population_eval_time",
    "validation_time",
    "workers",
    "pipe_exponent",
]


def save_training_history(history, path):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def build_genome_metadata(
    agent,
    validation_metrics=None,
    test_metrics=None,
    mutation_rate=None,
    mutation_strength=None,
):
    metadata = {
        "input_mode": agent.input_mode,
        "hidden_size": agent.hidden_size,
        "processed_input_size": agent.processed_input_size,
        "chromosome_length": agent.chromosome_length,
        "pipe_exponent": PIPE_EXPONENT,
    }
    if mutation_rate is not None:
        metadata["mutation_rate"] = mutation_rate
    if mutation_strength is not None:
        metadata["mutation_strength"] = mutation_strength
    if validation_metrics:
        metadata.update(validation_metrics)
    if test_metrics:
        metadata.update({f"test_{key}": value for key, value in test_metrics.items()})
    return metadata


def save_genome_metadata(metadata, path=BEST_METADATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


def load_genome_metadata(path=BEST_METADATA_PATH):
    with path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def save_progress_plots(history):
    if not history:
        return

    generations = [row["generation"] for row in history]
    best_fitness = [row["best_fitness"] for row in history]
    mean_fitness = [row["mean_fitness"] for row in history]
    validated_fitness = [row["validated_mean_fitness"] for row in history]
    best_pipes = [row["best_pipes"] for row in history]
    mean_pipes = [row["mean_pipes"] for row in history]
    validated_mean_pipes = [row["validated_mean_pipes"] for row in history]
    validated_min_pipes = [row["validated_min_pipes"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label="Generation Best Fitness", color="green")
    plt.plot(generations, mean_fitness, label="Mean Fitness", color="orange", linestyle="--")
    plt.plot(generations, validated_fitness, label="Validated Global Best Fitness", color="blue")
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
    plt.plot(
        generations,
        validated_mean_pipes,
        label="Validated Global Best Mean Pipes",
        color="green",
    )
    plt.plot(
        generations,
        validated_min_pipes,
        label="Validated Global Best Min Pipes",
        color="red",
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


def summarize_validation_results(results):
    pipes = np.array([result["pipes_passed"] for result in results], dtype=float)
    fitness = np.array([result["fitness"] for result in results], dtype=float)

    return {
        "validated_mean_pipes": float(np.mean(pipes)),
        "validated_min_pipes": int(np.min(pipes)),
        "validated_max_pipes": int(np.max(pipes)),
        "validated_std_pipes": float(np.std(pipes)),
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


def validate_candidates(
    population,
    seeds=VALIDATION_SEEDS,
    workers=1,
    executor=None,
    input_mode="raw180",
    hidden_size=16,
):
    grouped_results = evaluate_population_on_seed_batch(
        population,
        seeds=seeds,
        workers=workers,
        executor=executor,
        input_mode=input_mode,
        hidden_size=hidden_size,
    )
    return [summarize_validation_results(results) for results in grouped_results]


def validate_champion(
    chromosome,
    seeds=TEST_SEEDS,
    workers=1,
    executor=None,
    input_mode="raw180",
    hidden_size=16,
):
    results = evaluate_chromosome_on_seeds(
        chromosome,
        seeds=seeds,
        workers=workers,
        executor=executor,
        input_mode=input_mode,
        hidden_size=hidden_size,
    )
    return summarize_validation_results(results)


def default_worker_count():
    return max(1, (os.cpu_count() or 1) - 1)


def resolve_worker_count(requested_workers=None, no_parallel=False):
    if no_parallel:
        return 1
    if requested_workers is None:
        return default_worker_count()
    return max(1, requested_workers)


def parse_mutation_rate(value):
    try:
        mutation_rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "mutation rate must be a number in the range 0.0 < mutation_rate < 1.0"
        ) from exc

    if not 0.0 < mutation_rate < 1.0:
        raise argparse.ArgumentTypeError(
            "mutation rate must be in the range 0.0 < mutation_rate < 1.0"
        )

    return mutation_rate


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
    parser.add_argument(
        "--input-mode",
        choices=sorted(BirdAgent.INPUT_MODE_SIZES),
        default="raw180",
        help="Observation preprocessing mode.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=16,
        help="Number of hidden units in the controller network.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=parse_mutation_rate,
        default=None,
        help=(
            "Mutation probability for normal offspring. "
            f"Defaults to the configured value ({MUTATION_RATE})."
        ),
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
    input_mode="raw180",
    hidden_size=16,
    mutation_rate=None,
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
    if mutation_rate is None:
        mutation_rate = MUTATION_RATE
    else:
        mutation_rate = parse_mutation_rate(str(mutation_rate))

    agent_meta = BirdAgent(input_mode=input_mode, hidden_size=hidden_size)
    chromosome_length = agent_meta.chromosome_length

    BEST_GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)
    population = initialize_population(POPULATION_SIZE, chromosome_length)

    training_history = []
    global_best_genome = None
    global_best_result = None
    global_best_mutation_strength = None
    total_start_time = time.perf_counter()

    print(
        "Starting evolutionary training. "
        f"Population: {POPULATION_SIZE} | Generations: {generations} | "
        f"Workers: {workers} | Target pipes: {target_pipes} | "
        f"Input mode: {input_mode} | Hidden size: {hidden_size} | "
        f"Chromosome length: {chromosome_length} | "
        f"Mutation rate: {mutation_rate:.3f} | "
        f"Initial mutation strength: {INITIAL_MUTATION_STRENGTH:.3f} | "
        f"Validation seeds: {len(VALIDATION_SEEDS)} | "
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
            mutation_strength = adaptive_mutation_strength(generation_index, generations)
            generation_start_time = time.perf_counter()

            population_eval_start_time = time.perf_counter()
            evaluation_results = evaluate_population(
                population,
                workers=workers,
                executor=executor,
                input_mode=input_mode,
                hidden_size=hidden_size,
            )
            population_eval_time = time.perf_counter() - population_eval_start_time

            sorted_indices = sort_population_indices(evaluation_results)
            population = population[sorted_indices]
            evaluation_results = [evaluation_results[index] for index in sorted_indices]

            best_result = evaluation_results[0]
            population_summary = summarize_population(evaluation_results)

            validation_start_time = time.perf_counter()
            validation_count = min(TOP_VALIDATION_COUNT, len(population))
            validation_results = validate_candidates(
                population[:validation_count],
                seeds=VALIDATION_SEEDS,
                workers=workers,
                executor=executor,
                input_mode=input_mode,
                hidden_size=hidden_size,
            )
            for candidate_index, validation_result in enumerate(validation_results):
                if is_better_result(validation_result, global_best_result):
                    global_best_result = validation_result
                    global_best_genome = population[candidate_index].copy()
                    global_best_mutation_strength = mutation_strength
                    np.save(BEST_GENOME_PATH, global_best_genome)
                    save_genome_metadata(
                        build_genome_metadata(
                            agent_meta,
                            validation_metrics=global_best_result,
                            mutation_rate=mutation_rate,
                            mutation_strength=global_best_mutation_strength,
                        ),
                        BEST_METADATA_PATH,
                    )
                    print(
                        "New global best saved | "
                        f"Validated mean/min pipes: "
                        f"{global_best_result['validated_mean_pipes']:.2f}/"
                        f"{global_best_result['validated_min_pipes']} | "
                        f"Validated fitness: "
                        f"{global_best_result['validated_mean_fitness']:.1f}"
                    )

            if global_best_genome is not None:
                champion_result = validate_champion(
                    global_best_genome,
                    seeds=VALIDATION_SEEDS,
                    workers=workers,
                    executor=executor,
                    input_mode=input_mode,
                    hidden_size=hidden_size,
                )
                if is_better_result(champion_result, global_best_result):
                    global_best_result = champion_result
                    np.save(BEST_GENOME_PATH, global_best_genome)
                    save_genome_metadata(
                        build_genome_metadata(
                            agent_meta,
                            validation_metrics=global_best_result,
                            mutation_rate=mutation_rate,
                            mutation_strength=global_best_mutation_strength,
                        ),
                        BEST_METADATA_PATH,
                    )

            validation_time = time.perf_counter() - validation_start_time

            random_immigrant_ratio = RANDOM_IMMIGRANT_RATIO
            champion_offspring_ratio = CHAMPION_OFFSPRING_RATIO
            should_stop = should_stop_training(
                global_best_result,
                target_pipes=target_pipes,
                early_stop_on_target=early_stop_on_target,
            )

            if not should_stop:
                population = create_next_generation(
                    population,
                    chromosome_length=chromosome_length,
                    pop_size=POPULATION_SIZE,
                    mutation_strength=mutation_strength,
                    mutation_rate=mutation_rate,
                    global_best_genome=global_best_genome,
                    champion_offspring_ratio=champion_offspring_ratio,
                    random_immigrant_ratio=random_immigrant_ratio,
                )

            generation_time = time.perf_counter() - generation_start_time
            training_history.append(
                {
                    "generation": generation,
                    "best_fitness": best_result["fitness"],
                    "mean_fitness": population_summary["mean_fitness"],
                    "best_pipes": best_result["pipes_passed"],
                    "mean_pipes": population_summary["mean_pipes"],
                    "best_frames": best_result["frames"],
                    "mean_frames": population_summary["mean_frames"],
                    **global_best_result,
                    "mutation_rate": mutation_rate,
                    "mutation_strength": mutation_strength,
                    "random_immigrant_ratio": random_immigrant_ratio,
                    "champion_offspring_ratio": champion_offspring_ratio,
                    "generation_time": generation_time,
                    "population_eval_time": population_eval_time,
                    "validation_time": validation_time,
                    "workers": workers,
                    "pipe_exponent": PIPE_EXPONENT,
                }
            )

            print(
                f"Generation {generation:02d} | "
                f"Best fitness: {best_result['fitness']:.1f} | "
                f"Avg fitness: {population_summary['mean_fitness']:.1f} | "
                f"Best pipes: {best_result['pipes_passed']} | "
                f"Avg pipes: {population_summary['mean_pipes']:.2f} | "
                f"Validated global mean/min pipes: "
                f"{global_best_result['validated_mean_pipes']:.2f}/"
                f"{global_best_result['validated_min_pipes']} | "
                f"Mutation: {mutation_rate:.3f}/{mutation_strength:.3f} | "
                f"Champion offspring: {champion_offspring_ratio:.2f} | "
                f"Random immigrants: {random_immigrant_ratio:.2f}"
            )
            print(
                f"Generation {generation:02d} timing | "
                f"population={population_eval_time:.1f}s | "
                f"validation={validation_time:.1f}s | "
                f"total={generation_time:.1f}s | "
                f"workers={workers}"
            )

            if should_stop:
                print(
                    "Target reached by validated global best: "
                    f"{global_best_result['validated_mean_pipes']:.2f} mean pipes "
                    f"(target: {target_pipes})"
                )
                break

        if global_best_genome is not None:
            print("Evaluating saved champion on held-out test seeds...")
            test_result = validate_champion(
                global_best_genome,
                seeds=TEST_SEEDS,
                workers=workers,
                executor=executor,
                input_mode=input_mode,
                hidden_size=hidden_size,
            )
            save_genome_metadata(
                build_genome_metadata(
                    agent_meta,
                    validation_metrics=global_best_result,
                    test_metrics=test_result,
                    mutation_rate=mutation_rate,
                    mutation_strength=global_best_mutation_strength,
                ),
                BEST_METADATA_PATH,
            )
            print(
                "Test evaluation | "
                f"test_mean_pipes={test_result['validated_mean_pipes']:.2f} | "
                f"test_min_pipes={test_result['validated_min_pipes']} | "
                f"test_max_pipes={test_result['validated_max_pipes']} | "
                f"test_std_pipes={test_result['validated_std_pipes']:.2f} | "
                f"test_mean_fitness={test_result['validated_mean_fitness']:.1f}"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    save_training_history(training_history, TRAINING_HISTORY_PATH)
    save_progress_plots(training_history)

    print(f"\nTraining complete! Best genome saved as '{BEST_GENOME_PATH}'")
    print(f"Best genome metadata saved as '{BEST_METADATA_PATH}'")
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
        input_mode=args.input_mode,
        hidden_size=args.hidden_size,
        mutation_rate=args.mutation_rate,
    )


if __name__ == "__main__":
    main()

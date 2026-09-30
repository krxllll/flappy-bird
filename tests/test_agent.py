import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent  # noqa: E402
from flappy_bird_ai.genetic import (  # noqa: E402
    CHAMPION_OFFSPRING_RATIO,
    EARLY_STOP_ON_TARGET,
    RANDOM_IMMIGRANT_RATIO,
    TARGET_PIPES,
    TEST_SEEDS,
    VALIDATION_SEEDS,
    adaptive_mutation_strength,
    create_next_generation,
    is_better_result,
    sort_population_indices,
    target_reached,
)
from flappy_bird_ai.simulation import (  # noqa: E402
    PIPE_EXPONENT,
    PIPE_REWARD,
    calculate_fitness,
    evaluate_chromosome_on_seeds,
    evaluate_population,
    evaluate_population_on_seed_batch,
    suppress_gymnasium_observation_warnings,
)
from scripts import enjoy as enjoy_script  # noqa: E402
from scripts import train as train_script  # noqa: E402


def test_chromosome_length_is_2913():
    assert BirdAgent().chromosome_length == 2913


def test_binned18_hidden8_chromosome_length_is_161():
    assert BirdAgent(input_mode="binned18", hidden_size=8).chromosome_length == 161


def test_binned30_hidden8_chromosome_length_is_257():
    assert BirdAgent(input_mode="binned30", hidden_size=8).chromosome_length == 257


def test_preprocessing_returns_expected_feature_counts():
    observation = np.arange(180, dtype=float)

    assert len(BirdAgent(input_mode="raw180").preprocess_observation(observation)) == 180
    assert len(BirdAgent(input_mode="binned18").preprocess_observation(observation)) == 18
    assert len(BirdAgent(input_mode="binned30").preprocess_observation(observation)) == 30


def test_binned_preprocessing_uses_group_means():
    observation = np.arange(180, dtype=float)

    binned18 = BirdAgent(input_mode="binned18").preprocess_observation(observation)
    binned30 = BirdAgent(input_mode="binned30").preprocess_observation(observation)

    assert binned18[0] == np.mean(np.arange(10, dtype=float))
    assert binned30[0] == np.mean(np.arange(6, dtype=float))


def test_random_chromosome_maps_to_expected_weight_shapes():
    agent = BirdAgent()
    chromosome = np.random.uniform(-1.0, 1.0, agent.chromosome_length)

    W1, b1, W2, b2 = agent.map_chromosome_to_weights(chromosome)

    assert W1.shape == (180, 16)
    assert b1.shape == (1, 16)
    assert W2.shape == (16, 1)
    assert b2.shape == (1, 1)


def test_predict_returns_binary_action_for_random_observation():
    agent = BirdAgent()
    observation = np.random.uniform(-1.0, 1.0, (180,))
    chromosome = np.random.uniform(-1.0, 1.0, agent.chromosome_length)

    assert agent.predict(observation, chromosome) in (0, 1)


def test_predict_returns_binary_action_for_each_input_mode():
    observation = np.random.uniform(-1.0, 1.0, (180,))

    for input_mode in ("raw180", "binned18", "binned30"):
        agent = BirdAgent(input_mode=input_mode, hidden_size=8)
        chromosome = np.random.uniform(-1.0, 1.0, agent.chromosome_length)

        assert agent.predict(observation, chromosome) in (0, 1)


def test_invalid_chromosome_length_raises_clear_error():
    agent = BirdAgent(input_mode="binned18", hidden_size=8)

    with pytest.raises(ValueError, match="Chromosome length mismatch"):
        agent.map_chromosome_to_weights(np.zeros(agent.chromosome_length - 1))


def test_evaluation_result_shape_from_mocked_game(monkeypatch):
    class FakeActionSpace:
        def seed(self, seed):
            pass

    class FakeEnv:
        def __init__(self):
            self.step_count = 0
            self.action_space = FakeActionSpace()

        def reset(self, seed=None):
            return np.zeros(180), {}

        def step(self, action):
            self.step_count += 1
            terminated = self.step_count == 3
            score = 1 if self.step_count >= 2 else 0
            return np.zeros(180), 0.0, terminated, False, {"score": score}

        def close(self):
            pass

    monkeypatch.setattr("flappy_bird_ai.simulation.gym.make", lambda *args, **kwargs: FakeEnv())

    from flappy_bird_ai.simulation import evaluate_chromosome

    result = evaluate_chromosome(np.zeros(BirdAgent().chromosome_length))

    assert set(result) == {"fitness", "frames", "pipes_passed"}
    assert result["frames"] == 3
    assert result["pipes_passed"] == 1
    assert result["fitness"] == calculate_fitness(3, 1)


def test_evaluate_chromosome_accepts_seed(monkeypatch):
    seen = {"reset_seed": None, "action_seed": None}

    class FakeActionSpace:
        def seed(self, seed):
            seen["action_seed"] = seed

    class FakeEnv:
        def __init__(self):
            self.action_space = FakeActionSpace()

        def reset(self, seed=None):
            seen["reset_seed"] = seed
            return np.zeros(180), {}

        def step(self, action):
            return np.zeros(180), 0.0, True, False, {"score": 0}

        def close(self):
            pass

    monkeypatch.setattr("flappy_bird_ai.simulation.gym.make", lambda *args, **kwargs: FakeEnv())

    from flappy_bird_ai.simulation import evaluate_chromosome

    evaluate_chromosome(np.zeros(BirdAgent().chromosome_length), seed=123)

    assert seen["reset_seed"] == 123
    assert seen["action_seed"] == 123


def test_evaluate_chromosome_seed_does_not_require_action_space_seed(monkeypatch):
    class FakeActionSpace:
        pass

    class FakeEnv:
        def __init__(self):
            self.action_space = FakeActionSpace()

        def reset(self, seed=None):
            return np.zeros(180), {}

        def step(self, action):
            return np.zeros(180), 0.0, True, False, {"score": 0}

        def close(self):
            pass

    monkeypatch.setattr("flappy_bird_ai.simulation.gym.make", lambda *args, **kwargs: FakeEnv())

    from flappy_bird_ai.simulation import evaluate_chromosome

    assert evaluate_chromosome(np.zeros(BirdAgent().chromosome_length), seed=123)["pipes_passed"] == 0


def test_evaluate_chromosome_same_seed_is_deterministic(monkeypatch):
    class FakeActionSpace:
        def seed(self, seed):
            pass

    class FakeEnv:
        def __init__(self):
            self.step_count = 0
            self.limit = 1
            self.action_space = FakeActionSpace()

        def reset(self, seed=None):
            self.step_count = 0
            self.limit = int(np.random.randint(2, 5))
            return np.zeros(180), {}

        def step(self, action):
            self.step_count += 1
            score = int(np.random.randint(0, 4))
            terminated = self.step_count >= self.limit
            return np.zeros(180), 0.0, terminated, False, {"score": score}

        def close(self):
            pass

    monkeypatch.setattr("flappy_bird_ai.simulation.gym.make", lambda *args, **kwargs: FakeEnv())

    from flappy_bird_ai.simulation import evaluate_chromosome

    chromosome = np.zeros(BirdAgent().chromosome_length)

    assert evaluate_chromosome(chromosome, seed=321) == evaluate_chromosome(chromosome, seed=321)


def test_evaluate_population_workers_one_uses_sequential_path(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {"fitness": 1.0, "frames": 1, "pipes_passed": int(chromosome[0])}

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("ProcessPoolExecutor should not be used")

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)
    monkeypatch.setattr("flappy_bird_ai.simulation.ProcessPoolExecutor", FailingExecutor)

    results = evaluate_population(np.array([[0.0], [1.0]]), workers=1)

    assert [result["pipes_passed"] for result in results] == [0, 1]


def test_evaluate_population_can_reuse_existing_executor(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {"fitness": float(chromosome[0]), "frames": 1, "pipes_passed": int(chromosome[0])}

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("A new executor should not be created")

    class ExistingExecutor:
        def __init__(self):
            self.was_used = False

        def map(self, fn, chromosomes):
            self.was_used = True
            return [fn(chromosome) for chromosome in chromosomes]

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)
    monkeypatch.setattr("flappy_bird_ai.simulation.ProcessPoolExecutor", FailingExecutor)

    executor = ExistingExecutor()
    results = evaluate_population(np.array([[6.0], [7.0]]), workers=4, executor=executor)

    assert executor.was_used
    assert [result["pipes_passed"] for result in results] == [6, 7]


def test_seeded_population_batch_reuses_executor(monkeypatch):
    def fake_evaluate_chromosome(chromosome, render=False, frame_delay=0.0, seed=None):
        return {"fitness": float(seed), "frames": 1, "pipes_passed": int(chromosome[0] + seed)}

    class ExistingExecutor:
        def __init__(self):
            self.was_used = False

        def map(self, fn, jobs):
            self.was_used = True
            return [fn(job) for job in jobs]

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)

    executor = ExistingExecutor()
    grouped_results = evaluate_population_on_seed_batch(
        np.array([[1.0], [2.0]]),
        seeds=[11, 12],
        workers=4,
        executor=executor,
    )

    assert executor.was_used
    assert len(grouped_results) == 2
    assert [result["pipes_passed"] for result in grouped_results[0]] == [12, 13]


def test_evaluate_chromosome_on_seeds_groups_seeded_runs(monkeypatch):
    seen_seeds = []

    def fake_evaluate_chromosome(chromosome, render=False, frame_delay=0.0, seed=None):
        seen_seeds.append(seed)
        return {"fitness": float(seed), "frames": 1, "pipes_passed": int(seed)}

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)

    results = evaluate_chromosome_on_seeds(np.array([1.0]), seeds=[5, 6], workers=1)

    assert seen_seeds == [5, 6]
    assert [result["pipes_passed"] for result in results] == [5, 6]


def test_suppress_gymnasium_observation_warnings_matches_warn_prefix():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        suppress_gymnasium_observation_warnings()
        warnings.warn(
            "WARN: The obs returned by the `reset()` method is not within the observation space.",
            UserWarning,
        )
        warnings.warn("A different user warning", UserWarning)

    assert len(captured) == 1
    assert str(captured[0].message) == "A different user warning"


def test_fitness_increases_significantly_when_pipes_increase():
    no_pipe_fitness = calculate_fitness(frames_survived=300, pipes_passed=0)
    one_pipe_fitness = calculate_fitness(frames_survived=300, pipes_passed=1)

    assert one_pipe_fitness - no_pipe_fitness == PIPE_REWARD


def test_fitness_uses_configurable_pipe_exponent():
    fitness = calculate_fitness(
        frames_survived=100,
        pipes_passed=4,
        pipe_reward=1000,
        pipe_exponent=PIPE_EXPONENT,
    )

    assert PIPE_EXPONENT == 1.75
    assert fitness == 100 + (4**1.75) * 1000


def test_higher_pipe_count_beats_higher_survival_time():
    long_survival = calculate_fitness(frames_survived=3000, pipes_passed=1)
    more_pipes = calculate_fitness(frames_survived=100, pipes_passed=3)

    assert more_pipes > long_survival


def test_population_sorting_prioritizes_pipes_over_fitness():
    evaluation_results = [
        {"pipes_passed": 1, "fitness": 50000.0, "frames": 900},
        {"pipes_passed": 3, "fitness": 10000.0, "frames": 200},
        {"pipes_passed": 3, "fitness": 11000.0, "frames": 100},
        {"pipes_passed": 3, "fitness": 11000.0, "frames": 300},
    ]

    assert sort_population_indices(evaluation_results) == [3, 2, 1, 0]


def test_population_sorting_uses_mean_pipes_when_present():
    evaluation_results = [
        {"mean_pipes": 2.0, "pipes_passed": 9, "fitness": 100.0, "frames": 1},
        {"mean_pipes": 3.0, "pipes_passed": 1, "fitness": 50.0, "frames": 1},
    ]

    assert sort_population_indices(evaluation_results) == [1, 0]


def test_champion_comparison_uses_validated_mean_min_then_fitness():
    current_best = {
        "validated_mean_pipes": 3.0,
        "validated_min_pipes": 2,
        "validated_mean_fitness": 12000.0,
    }
    higher_mean = {
        "validated_mean_pipes": 4.0,
        "validated_min_pipes": 0,
        "validated_mean_fitness": 1000.0,
    }
    higher_min = {
        "validated_mean_pipes": 3.0,
        "validated_min_pipes": 3,
        "validated_mean_fitness": 1000.0,
    }
    higher_fitness = {
        "validated_mean_pipes": 3.0,
        "validated_min_pipes": 2,
        "validated_mean_fitness": 13000.0,
    }

    assert is_better_result(higher_mean, current_best)
    assert is_better_result(higher_min, current_best)
    assert is_better_result(higher_fitness, current_best)


def test_target_pipe_threshold_detects_success():
    assert target_reached({"pipes_passed": 10}, target_pipes=10)
    assert not target_reached({"pipes_passed": 9}, target_pipes=10)
    assert target_reached({"validated_mean_pipes": 10.0}, target_pipes=10)
    assert not target_reached({"validated_mean_pipes": 9.9}, target_pipes=10)


def test_no_early_stop_disables_target_stop():
    assert not train_script.should_stop_training(
        {"validated_mean_pipes": TARGET_PIPES},
        target_pipes=TARGET_PIPES,
        early_stop_on_target=False,
    )


def test_target_pipes_argument_overrides_default_target():
    args = train_script.parse_args(["--target-pipes", "20"])

    assert args.target_pipes == 20
    assert not train_script.should_stop_training(
        {"validated_mean_pipes": 10.5},
        target_pipes=args.target_pipes,
        early_stop_on_target=True,
    )
    assert train_script.should_stop_training(
        {"validated_mean_pipes": 20.0},
        target_pipes=args.target_pipes,
        early_stop_on_target=True,
    )


def test_default_early_stop_behavior_remains_unchanged():
    args = train_script.parse_args([])

    assert args.no_early_stop is False
    assert args.target_pipes is None
    assert train_script.should_stop_training(
        {"validated_mean_pipes": TARGET_PIPES},
        target_pipes=TARGET_PIPES,
        early_stop_on_target=EARLY_STOP_ON_TARGET,
    )


def test_worker_arguments_resolve_to_expected_modes(monkeypatch):
    monkeypatch.setattr(train_script.os, "cpu_count", lambda: 8)

    assert train_script.resolve_worker_count(None, no_parallel=False) == 7
    assert train_script.resolve_worker_count(4, no_parallel=False) == 4
    assert train_script.resolve_worker_count(4, no_parallel=True) == 1
    assert train_script.resolve_worker_count(0, no_parallel=False) == 1


def test_generation_argument_is_parsed():
    args = train_script.parse_args(["--generations", "3", "--workers", "2", "--no-parallel"])

    assert args.generations == 3
    assert args.workers == 2
    assert args.no_parallel is True


def test_architecture_arguments_are_parsed():
    args = train_script.parse_args(["--input-mode", "binned18", "--hidden-size", "8"])

    assert args.input_mode == "binned18"
    assert args.hidden_size == 8


def test_default_mutation_rate_argument_preserves_configured_default():
    args = train_script.parse_args([])

    assert args.mutation_rate is None


def test_mutation_rate_argument_overrides_default():
    args = train_script.parse_args(["--mutation-rate", "0.10"])

    assert args.mutation_rate == 0.10


def test_invalid_mutation_rate_arguments_are_rejected():
    for value in ("0", "1", "-0.1", "1.2", "not-a-number"):
        with pytest.raises(SystemExit):
            train_script.parse_args(["--mutation-rate", value])


def test_test_seeds_are_separate_from_validation_seeds():
    assert set(TEST_SEEDS).isdisjoint(VALIDATION_SEEDS)


def test_adaptive_mutation_strength_decreases_to_minimum():
    start_strength = adaptive_mutation_strength(
        generation_index=0,
        total_generations=50,
        initial_strength=0.2,
        min_strength=0.05,
    )
    late_strength = adaptive_mutation_strength(
        generation_index=49,
        total_generations=50,
        initial_strength=0.2,
        min_strength=0.05,
    )
    beyond_end_strength = adaptive_mutation_strength(
        generation_index=100,
        total_generations=50,
        initial_strength=0.2,
        min_strength=0.05,
    )

    assert start_strength == 0.2
    assert late_strength < start_strength
    assert beyond_end_strength == 0.05


def test_elites_are_copied_and_not_mutated_in_place():
    population = np.array(
        [
            np.full(5, 10.0),
            np.full(5, 5.0),
            np.full(5, 1.0),
            np.full(5, -1.0),
        ]
    )

    next_generation = create_next_generation(
        population,
        pop_size=4,
        elite_size=2,
        mutation_strength=1.0,
    )

    assert np.array_equal(next_generation[0], population[0])
    assert np.array_equal(next_generation[1], population[1])
    assert not np.shares_memory(next_generation[0], population[0])
    assert not np.shares_memory(next_generation[1], population[1])


def test_global_best_is_injected_into_next_generation():
    population = np.array(
        [
            np.full(5, 3.0),
            np.full(5, 2.0),
            np.full(5, 1.0),
        ]
    )
    global_best = np.full(5, 99.0)

    next_generation = create_next_generation(
        population,
        pop_size=5,
        elite_size=1,
        mutation_rate=0.0,
        global_best_genome=global_best,
        champion_offspring_ratio=0.0,
    )

    assert np.array_equal(next_generation[0], global_best)
    assert not np.shares_memory(next_generation[0], global_best)


def test_champion_offspring_use_small_mutation_without_mutating_original(monkeypatch):
    import flappy_bird_ai.genetic as genetic

    mutation_calls = []

    def fake_mutate(chromosome, mutation_strength, mutation_rate):
        mutation_calls.append((mutation_strength, mutation_rate, chromosome.copy()))
        chromosome += 1.0
        return chromosome

    monkeypatch.setattr(genetic, "mutate", fake_mutate)

    global_best = np.full(5, 42.0)
    genetic.create_next_generation(
        np.array([np.full(5, 3.0), np.full(5, 2.0), np.full(5, 1.0)]),
        pop_size=5,
        elite_size=0,
        mutation_rate=0.15,
        mutation_strength=0.2,
        global_best_genome=global_best,
        champion_offspring_ratio=0.4,
        random_immigrant_ratio=0.0,
    )

    assert np.array_equal(global_best, np.full(5, 42.0))
    assert mutation_calls[0][0] == 0.02
    assert mutation_calls[0][1] == 0.03


def test_normal_offspring_use_passed_mutation_rate(monkeypatch):
    import flappy_bird_ai.genetic as genetic

    mutation_rates = []

    def fake_mutate(chromosome, mutation_strength, mutation_rate):
        mutation_rates.append(mutation_rate)
        return chromosome

    monkeypatch.setattr(genetic, "mutate", fake_mutate)

    genetic.create_next_generation(
        np.array([np.full(5, 3.0), np.full(5, 2.0), np.full(5, 1.0)]),
        pop_size=3,
        elite_size=0,
        mutation_rate=0.19,
        mutation_strength=0.2,
        champion_offspring_ratio=0.0,
        random_immigrant_ratio=0.0,
    )

    assert mutation_rates
    assert set(mutation_rates) == {0.19}


def test_random_immigrants_preserve_population_size():
    next_generation = create_next_generation(
        np.array([np.full(5, 3.0), np.full(5, 2.0), np.full(5, 1.0)]),
        pop_size=10,
        elite_size=1,
        mutation_rate=0.0,
        random_immigrant_ratio=RANDOM_IMMIGRANT_RATIO,
    )

    assert next_generation.shape == (10, 5)


def test_training_loop_writes_simple_history_and_updates_global_best(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeAgent:
        def __init__(self, input_mode="raw180", hidden_size=16):
            self.input_mode = input_mode
            self.hidden_size = hidden_size
            self.processed_input_size = 1
            self.chromosome_length = 1

    saved_next_generation = {}

    def fake_initialize_population(pop_size, chromosome_length):
        return np.array([[1.0], [2.0], [3.0]])

    def fake_evaluate_population(
        population,
        workers=1,
        executor=None,
        input_mode="raw180",
        hidden_size=16,
    ):
        return [
            {"fitness": float(chromosome[0] * 100), "frames": int(chromosome[0]), "pipes_passed": int(chromosome[0])}
            for chromosome in population
        ]

    def fake_evaluate_population_on_seed_batch(
        population,
        seeds,
        workers=1,
        executor=None,
        input_mode="raw180",
        hidden_size=16,
    ):
        grouped = []
        for chromosome in population:
            value = int(chromosome[0])
            pipes = [4, 4] if value == 2 else [value, value]
            grouped.append(
                [
                    {"fitness": float(pipe * 100), "frames": pipe, "pipes_passed": pipe}
                    for pipe in pipes
                ]
            )
        return grouped

    def fake_evaluate_chromosome_on_seeds(
        chromosome,
        seeds,
        workers=1,
        executor=None,
        input_mode="raw180",
        hidden_size=16,
    ):
        value = int(chromosome[0])
        pipes = [4, 4] if value == 2 else [value, value]
        return [
            {"fitness": float(pipe * 100), "frames": pipe, "pipes_passed": pipe}
            for pipe in pipes
        ]

    def fake_create_next_generation(population, **kwargs):
        saved_next_generation["global_best_genome"] = kwargs["global_best_genome"].copy()
        saved_next_generation["mutation_rate"] = kwargs["mutation_rate"]
        saved_next_generation["mutation_strength"] = kwargs["mutation_strength"]
        return population

    monkeypatch.setattr(train_script, "BirdAgent", FakeAgent)
    monkeypatch.setattr(train_script, "POPULATION_SIZE", 3)
    monkeypatch.setattr(train_script, "TOP_VALIDATION_COUNT", 3)
    monkeypatch.setattr(train_script, "VALIDATION_SEEDS", [10, 11])
    monkeypatch.setattr(train_script, "TEST_SEEDS", [20, 21])
    monkeypatch.setattr(train_script, "BEST_GENOME_PATH", tmp_path / "best.npy")
    monkeypatch.setattr(train_script, "BEST_METADATA_PATH", tmp_path / "best_metadata.json")
    monkeypatch.setattr(train_script, "TRAINING_HISTORY_PATH", tmp_path / "history.csv")
    monkeypatch.setattr(train_script, "FITNESS_CHART_PATH", tmp_path / "fitness.png")
    monkeypatch.setattr(train_script, "PIPE_CHART_PATH", tmp_path / "pipes.png")
    monkeypatch.setattr(train_script, "initialize_population", fake_initialize_population)
    monkeypatch.setattr(train_script, "evaluate_population", fake_evaluate_population)
    monkeypatch.setattr(train_script, "evaluate_population_on_seed_batch", fake_evaluate_population_on_seed_batch)
    monkeypatch.setattr(train_script, "evaluate_chromosome_on_seeds", fake_evaluate_chromosome_on_seeds)
    monkeypatch.setattr(train_script, "create_next_generation", fake_create_next_generation)
    monkeypatch.setattr(train_script, "save_progress_plots", lambda history: None)

    train_script.train_evolutionary_ai(
        workers=1,
        generations=2,
        target_pipes=10,
        early_stop_on_target=False,
        mutation_rate=0.12,
    )
    captured = capsys.readouterr()

    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert reader.fieldnames == train_script.CSV_FIELDNAMES
    assert len(rows) == 2
    assert rows[0]["validated_mean_pipes"] == "4.0"
    assert rows[0]["validated_min_pipes"] == "4"
    assert rows[0]["mutation_rate"] == "0.12"
    assert np.array_equal(saved_next_generation["global_best_genome"], np.array([2.0]))
    assert saved_next_generation["mutation_rate"] == 0.12
    assert np.array_equal(np.load(tmp_path / "best.npy"), np.array([2.0]))
    assert "Mutation rate: 0.120" in captured.out
    assert "Mutation: 0.120/" in captured.out

    metadata = train_script.load_genome_metadata(tmp_path / "best_metadata.json")
    assert metadata["input_mode"] == "raw180"
    assert metadata["hidden_size"] == 16
    assert metadata["processed_input_size"] == 1
    assert metadata["chromosome_length"] == 1
    assert metadata["pipe_exponent"] == PIPE_EXPONENT
    assert metadata["mutation_rate"] == 0.12
    assert metadata["mutation_strength"] == train_script.INITIAL_MUTATION_STRENGTH
    assert metadata["mutation_strength"] > saved_next_generation["mutation_strength"]
    assert metadata["validated_mean_pipes"] == 4.0
    assert metadata["test_validated_mean_pipes"] == 4.0


def test_metadata_save_load_round_trip(tmp_path):
    metadata = {
        "input_mode": "binned18",
        "hidden_size": 8,
        "processed_input_size": 18,
        "chromosome_length": 161,
        "pipe_exponent": PIPE_EXPONENT,
        "mutation_rate": 0.1,
        "mutation_strength": 0.2,
        "validated_mean_pipes": 7.0,
    }

    metadata_path = tmp_path / "metadata.json"
    train_script.save_genome_metadata(metadata, metadata_path)

    assert train_script.load_genome_metadata(metadata_path) == metadata


def test_enjoy_constructs_agent_config_from_metadata(monkeypatch, tmp_path):
    genome_path = tmp_path / "best.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(genome_path, np.zeros(161))
    metadata_path.write_text(
        json.dumps({"input_mode": "binned18", "hidden_size": 8}),
        encoding="utf-8",
    )
    seen = {}

    def fake_evaluate_chromosome(
        chromosome,
        render=False,
        frame_delay=0.0,
        seed=None,
        input_mode="raw180",
        hidden_size=16,
    ):
        seen["input_mode"] = input_mode
        seen["hidden_size"] = hidden_size
        seen["chromosome_length"] = len(chromosome)
        return {"fitness": 1.0, "frames": 1, "pipes_passed": 0}

    monkeypatch.setattr(enjoy_script, "evaluate_chromosome", fake_evaluate_chromosome)
    monkeypatch.setattr(enjoy_script, "suppress_gymnasium_observation_warnings", lambda: None)

    enjoy_script.watch_best_bird(genome_path=genome_path, metadata_path=metadata_path)

    assert seen == {
        "input_mode": "binned18",
        "hidden_size": 8,
        "chromosome_length": 161,
    }


def test_plot_lines_are_simple(monkeypatch):
    labels = []

    monkeypatch.setattr(train_script.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "plot", lambda *args, **kwargs: labels.append(kwargs["label"]))
    monkeypatch.setattr(train_script.plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script.plt, "close", lambda *args, **kwargs: None)

    train_script.save_progress_plots(
        [
            {
                "generation": 1,
                "best_fitness": 10.0,
                "mean_fitness": 5.0,
                "validated_mean_fitness": 8.0,
                "best_pipes": 2,
                "mean_pipes": 1.0,
                "validated_mean_pipes": 1.5,
                "validated_min_pipes": 1,
            }
        ]
    )

    assert labels == [
        "Generation Best Fitness",
        "Mean Fitness",
        "Validated Global Best Fitness",
        "Generation Best Pipes",
        "Mean Pipes",
        "Validated Global Best Mean Pipes",
        "Validated Global Best Min Pipes",
    ]

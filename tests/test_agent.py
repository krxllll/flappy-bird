import sys
import warnings
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent  # noqa: E402
from flappy_bird_ai.genetic import (  # noqa: E402
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
    evaluate_chromosome_many,
    evaluate_population,
    evaluate_population_on_seed_batch,
    reevaluate_candidates,
    suppress_gymnasium_observation_warnings,
)
from scripts import train as train_script  # noqa: E402


def test_chromosome_length_is_2913():
    agent = BirdAgent()

    assert agent.chromosome_length == 2913


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

    action = agent.predict(observation, chromosome)

    assert action in (0, 1)


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
            observation = np.zeros(180)
            terminated = self.step_count == 3
            info = {"score": 1 if self.step_count >= 2 else 0}
            return observation, 0.0, terminated, False, info

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

    result = evaluate_chromosome(np.zeros(BirdAgent().chromosome_length), seed=123)

    assert result["pipes_passed"] == 0


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


def test_evaluate_population_returns_results_in_sequential_mode(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        pipes = int(chromosome[0])
        return {"fitness": float(pipes * 100), "frames": 10, "pipes_passed": pipes}

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)

    results = evaluate_population(np.array([[1.0], [2.0], [3.0]]), workers=1)

    assert [result["pipes_passed"] for result in results] == [1, 2, 3]


def test_evaluate_population_workers_one_uses_sequential_path(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {"fitness": 1.0, "frames": 1, "pipes_passed": 0}

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("ProcessPoolExecutor should not be used")

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)
    monkeypatch.setattr("flappy_bird_ai.simulation.ProcessPoolExecutor", FailingExecutor)

    results = evaluate_population(np.array([[0.0], [1.0]]), workers=1)

    assert len(results) == 2


def test_suppress_gymnasium_observation_warnings_helper_exists():
    assert callable(suppress_gymnasium_observation_warnings)


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


def test_evaluate_population_worker_count_is_configurable(monkeypatch):
    seen_workers = []

    def fake_evaluate_chromosome(chromosome):
        return {"fitness": float(chromosome[0]), "frames": 1, "pipes_passed": int(chromosome[0])}

    class FakeExecutor:
        def __init__(self, max_workers):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, fn, chromosomes):
            return [fn(chromosome) for chromosome in chromosomes]

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)
    monkeypatch.setattr("flappy_bird_ai.simulation.ProcessPoolExecutor", FakeExecutor)

    results = evaluate_population(np.array([[4.0], [5.0]]), workers=4)

    assert seen_workers == [4]
    assert [result["pipes_passed"] for result in results] == [4, 5]


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


def test_evaluate_chromosome_many_returns_expected_episode_count(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {"fitness": float(chromosome[0]), "frames": 1, "pipes_passed": int(chromosome[0])}

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)

    results = evaluate_chromosome_many(np.array([3.0]), episodes=4, workers=1)

    assert len(results) == 4
    assert all(result["pipes_passed"] == 3 for result in results)


def test_reevaluate_candidates_accepts_existing_executor(monkeypatch):
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

    grouped_results = reevaluate_candidates(
        np.array([[2.0], [5.0]]),
        episodes=3,
        workers=4,
        executor=ExistingExecutor(),
    )

    assert len(grouped_results) == 2
    assert len(grouped_results[0]) == 3
    assert grouped_results[1][0]["pipes_passed"] == 5


def test_fixed_seed_batch_is_reused_for_all_chromosomes(monkeypatch):
    seen_jobs = []

    def fake_evaluate_chromosome(chromosome, render=False, frame_delay=0.0, seed=None):
        seen_jobs.append((int(chromosome[0]), seed))
        return {"fitness": float(seed), "frames": 1, "pipes_passed": int(seed)}

    monkeypatch.setattr("flappy_bird_ai.simulation.evaluate_chromosome", fake_evaluate_chromosome)

    grouped_results = evaluate_population_on_seed_batch(
        np.array([[1.0], [2.0]]),
        seeds=[11, 12, 13],
        workers=1,
    )

    assert seen_jobs == [(1, 11), (1, 12), (1, 13), (2, 11), (2, 12), (2, 13)]
    assert len(grouped_results) == 2
    assert [result["pipes_passed"] for result in grouped_results[0]] == [11, 12, 13]


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


def test_global_best_comparison_uses_pipes_then_fitness_then_frames():
    current_best = {"pipes_passed": 3, "fitness": 12000.0, "frames": 100}
    more_pipes = {"pipes_passed": 4, "fitness": 1000.0, "frames": 50}
    same_pipes_more_fitness = {"pipes_passed": 3, "fitness": 13000.0, "frames": 10}
    same_pipes_less_fitness = {"pipes_passed": 3, "fitness": 11000.0, "frames": 500}

    assert is_better_result(more_pipes, current_best)
    assert is_better_result(same_pipes_more_fitness, current_best)
    assert not is_better_result(same_pipes_less_fitness, current_best)


def test_global_best_comparison_accepts_validated_summary_fields():
    current_best = {
        "validated_mean_pipes": 2.0,
        "validated_mean_fitness": 12000.0,
        "mean_frames": 100.0,
    }
    candidate = {
        "validated_mean_pipes": 2.5,
        "validated_mean_fitness": 9000.0,
        "mean_frames": 50.0,
    }

    assert is_better_result(candidate, current_best)


def test_target_pipe_threshold_detects_success():
    assert target_reached({"pipes_passed": 10}, target_pipes=10)
    assert not target_reached({"pipes_passed": 9}, target_pipes=10)
    assert target_reached({"validated_mean_pipes": 10.0}, target_pipes=10)
    assert not target_reached({"validated_mean_pipes": 9.9}, target_pipes=10)


def test_no_early_stop_disables_target_stop():
    result = {"validated_mean_pipes": TARGET_PIPES}

    assert not train_script.should_stop_training(
        result,
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


def test_champion_offspring_are_copied_from_global_best_safely():
    population = np.array(
        [
            np.full(5, 3.0),
            np.full(5, 2.0),
            np.full(5, 1.0),
        ]
    )
    global_best = np.full(5, 42.0)

    next_generation = create_next_generation(
        population,
        pop_size=6,
        elite_size=0,
        champion_mutation_rate=0.0,
        global_best_genome=global_best,
        champion_offspring_ratio=0.5,
    )

    assert np.array_equal(next_generation[0], global_best)
    assert np.array_equal(next_generation[1], global_best)
    assert np.array_equal(next_generation[2], global_best)
    assert np.array_equal(next_generation[3], global_best)
    assert all(not np.shares_memory(next_generation[index], global_best) for index in range(4))


def test_champion_offspring_use_small_mutation_without_mutating_original(monkeypatch):
    import flappy_bird_ai.genetic as genetic

    mutation_calls = []

    def fake_mutate(chromosome, mutation_strength, mutation_rate):
        mutation_calls.append((mutation_strength, mutation_rate, chromosome.copy()))
        chromosome += 1.0
        return chromosome

    monkeypatch.setattr(genetic, "mutate", fake_mutate)

    population = np.array(
        [
            np.full(5, 3.0),
            np.full(5, 2.0),
            np.full(5, 1.0),
        ]
    )
    global_best = np.full(5, 42.0)

    genetic.create_next_generation(
        population,
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


def test_random_immigrants_preserve_population_size():
    population = np.array(
        [
            np.full(5, 3.0),
            np.full(5, 2.0),
            np.full(5, 1.0),
        ]
    )

    next_generation = create_next_generation(
        population,
        pop_size=10,
        elite_size=1,
        mutation_rate=0.0,
        random_immigrant_ratio=RANDOM_IMMIGRANT_RATIO,
    )

    assert next_generation.shape == (10, 5)

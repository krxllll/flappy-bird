import sys
import warnings
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent  # noqa: E402
from flappy_bird_ai.genetic import (  # noqa: E402
    adapt_mutation_after_breakthrough,
    adaptive_mutation_strength,
    apply_patience_diversity,
    calculate_champion_score,
    create_next_generation,
    get_champion_offspring_ratio,
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
    reevaluate_candidates,
    suppress_gymnasium_observation_warnings,
)


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
    class FakeEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self):
            return np.zeros(180), {}

        def step(self, action):
            self.step_count += 1
            observation = np.zeros(180)
            reward = 0.0
            terminated = self.step_count == 3
            truncated = False
            info = {"score": 1 if self.step_count >= 2 else 0}
            return observation, reward, terminated, truncated, info

        def close(self):
            pass

    def fake_make(env_name, render_mode=None):
        return FakeEnv()

    monkeypatch.setattr("flappy_bird_ai.simulation.gym.make", fake_make)

    from flappy_bird_ai.simulation import evaluate_chromosome

    agent = BirdAgent()
    chromosome = np.zeros(agent.chromosome_length)
    result = evaluate_chromosome(chromosome)

    assert set(result) == {"fitness", "frames", "pipes_passed"}
    assert result["frames"] == 3
    assert result["pipes_passed"] == 1
    assert result["fitness"] == calculate_fitness(3, 1)


def test_evaluate_population_returns_results_in_sequential_mode(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        pipes = int(chromosome[0])
        return {
            "fitness": float(pipes * 100),
            "frames": 10,
            "pipes_passed": pipes,
        }

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )

    population = np.array([[1.0], [2.0], [3.0]])
    results = evaluate_population(population, workers=1)

    assert results == [
        {"fitness": 100.0, "frames": 10, "pipes_passed": 1},
        {"fitness": 200.0, "frames": 10, "pipes_passed": 2},
        {"fitness": 300.0, "frames": 10, "pipes_passed": 3},
    ]


def test_evaluate_population_workers_one_uses_sequential_path(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {"fitness": 1.0, "frames": 1, "pipes_passed": 0}

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("ProcessPoolExecutor should not be used")

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )
    monkeypatch.setattr(
        "flappy_bird_ai.simulation.ProcessPoolExecutor",
        FailingExecutor,
    )

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
        return {
            "fitness": float(chromosome[0]),
            "frames": 1,
            "pipes_passed": int(chromosome[0]),
        }

    class FakeExecutor:
        def __init__(self, max_workers):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, fn, chromosomes):
            return [fn(chromosome) for chromosome in chromosomes]

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )
    monkeypatch.setattr(
        "flappy_bird_ai.simulation.ProcessPoolExecutor",
        FakeExecutor,
    )

    results = evaluate_population(np.array([[4.0], [5.0]]), workers=4)

    assert seen_workers == [4]
    assert results[0]["pipes_passed"] == 4
    assert results[1]["pipes_passed"] == 5


def test_evaluate_population_can_reuse_existing_executor(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {
            "fitness": float(chromosome[0]),
            "frames": 1,
            "pipes_passed": int(chromosome[0]),
        }

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("A new executor should not be created")

    class ExistingExecutor:
        def __init__(self):
            self.was_used = False

        def map(self, fn, chromosomes):
            self.was_used = True
            return [fn(chromosome) for chromosome in chromosomes]

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )
    monkeypatch.setattr(
        "flappy_bird_ai.simulation.ProcessPoolExecutor",
        FailingExecutor,
    )

    executor = ExistingExecutor()
    results = evaluate_population(
        np.array([[6.0], [7.0]]),
        workers=4,
        executor=executor,
    )

    assert executor.was_used
    assert results[0]["pipes_passed"] == 6
    assert results[1]["pipes_passed"] == 7


def test_evaluate_chromosome_many_returns_expected_episode_count(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {
            "fitness": float(chromosome[0]),
            "frames": 1,
            "pipes_passed": int(chromosome[0]),
        }

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )

    results = evaluate_chromosome_many(np.array([3.0]), episodes=4, workers=1)

    assert len(results) == 4
    assert all(result["pipes_passed"] == 3 for result in results)


def test_reevaluate_candidates_accepts_existing_executor(monkeypatch):
    def fake_evaluate_chromosome(chromosome):
        return {
            "fitness": float(chromosome[0]),
            "frames": 1,
            "pipes_passed": int(chromosome[0]),
        }

    class FailingExecutor:
        def __init__(self, max_workers):
            raise AssertionError("A new executor should not be created")

    class ExistingExecutor:
        def __init__(self):
            self.was_used = False

        def map(self, fn, chromosomes):
            self.was_used = True
            return [fn(chromosome) for chromosome in chromosomes]

    monkeypatch.setattr(
        "flappy_bird_ai.simulation.evaluate_chromosome",
        fake_evaluate_chromosome,
    )
    monkeypatch.setattr(
        "flappy_bird_ai.simulation.ProcessPoolExecutor",
        FailingExecutor,
    )

    executor = ExistingExecutor()
    grouped_results = reevaluate_candidates(
        np.array([[2.0], [5.0]]),
        episodes=3,
        workers=4,
        executor=executor,
    )

    assert executor.was_used
    assert len(grouped_results) == 2
    assert len(grouped_results[0]) == 3
    assert len(grouped_results[1]) == 3
    assert grouped_results[0][0]["pipes_passed"] == 2
    assert grouped_results[1][0]["pipes_passed"] == 5


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


def test_target_pipe_threshold_detects_success():
    assert target_reached({"pipes_passed": 10}, target_pipes=10)
    assert not target_reached({"pipes_passed": 9}, target_pipes=10)
    assert target_reached({"validated_mean_pipes": 10.0}, target_pipes=10)
    assert not target_reached({"validated_mean_pipes": 9.9}, target_pipes=10)


def test_champion_comparison_prioritizes_validated_mean_pipes():
    current_best = {
        "validated_mean_pipes": 4.0,
        "validated_mean_fitness": 25000.0,
        "champion_score": 4.8,
    }
    candidate = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 20000.0,
        "champion_score": 4.2,
    }

    assert is_better_result(candidate, current_best)


def test_population_sorting_prioritizes_pipes_over_fitness():
    evaluation_results = [
        {"pipes_passed": 1, "fitness": 50000.0, "frames": 900},
        {"pipes_passed": 3, "fitness": 10000.0, "frames": 200},
        {"pipes_passed": 3, "fitness": 11000.0, "frames": 100},
        {"pipes_passed": 3, "fitness": 11000.0, "frames": 300},
    ]

    assert sort_population_indices(evaluation_results) == [3, 2, 1, 0]


def test_champion_score_breaks_ties_for_equal_mean_pipes():
    current_best = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 25000.0,
        "champion_score": 4.8,
    }
    candidate = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 20000.0,
        "champion_score": 5.1,
    }

    assert is_better_result(candidate, current_best)


def test_validated_mean_fitness_is_tertiary_tiebreaker():
    current_best = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 20000.0,
        "champion_score": 5.1,
    }
    candidate = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 22000.0,
        "champion_score": 5.1,
    }

    assert is_better_result(candidate, current_best)


def test_champion_comparison_prefers_stable_higher_min_pipes():
    stable_score = calculate_champion_score(
        validated_mean_pipes=10.0,
        validated_min_pipes=8,
        validated_std_pipes=1.0,
    )
    unstable_score = calculate_champion_score(
        validated_mean_pipes=10.0,
        validated_min_pipes=2,
        validated_std_pipes=5.0,
    )
    stable = {
        "validated_mean_pipes": 10.0,
        "validated_mean_fitness": 10000.0,
        "champion_score": stable_score,
    }
    unstable = {
        "validated_mean_pipes": 10.0,
        "validated_mean_fitness": 10000.0,
        "champion_score": unstable_score,
    }

    assert stable_score > unstable_score
    assert is_better_result(stable, unstable)


def test_lucky_high_max_low_mean_candidate_does_not_replace_stable_champion():
    stable = {
        "validated_mean_pipes": 8.0,
        "validated_min_pipes": 7,
        "validated_max_pipes": 10,
        "validated_std_pipes": 1.0,
        "validated_mean_fitness": 9000.0,
    }
    lucky = {
        "validated_mean_pipes": 5.0,
        "validated_min_pipes": 0,
        "validated_max_pipes": 30,
        "validated_std_pipes": 9.0,
        "validated_mean_fitness": 20000.0,
    }
    stable["champion_score"] = calculate_champion_score(
        stable["validated_mean_pipes"],
        stable["validated_min_pipes"],
        stable["validated_std_pipes"],
    )
    lucky["champion_score"] = calculate_champion_score(
        lucky["validated_mean_pipes"],
        lucky["validated_min_pipes"],
        lucky["validated_std_pipes"],
    )

    assert not is_better_result(lucky, stable)


def test_stable_low_pipe_candidate_does_not_replace_higher_pipe_champion():
    higher_pipe_champion = {
        "validated_mean_pipes": 6.0,
        "validated_mean_fitness": 10000.0,
        "champion_score": 5.0,
    }
    stable_low_pipe_candidate = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 30000.0,
        "champion_score": 8.0,
    }

    assert not is_better_result(stable_low_pipe_candidate, higher_pipe_champion)


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


def test_mutation_is_reduced_after_breakthrough_threshold():
    unchanged_rate, unchanged_strength = adapt_mutation_after_breakthrough(
        mutation_rate=0.15,
        mutation_strength=0.12,
        global_best_pipes=9,
    )
    reduced_rate, reduced_strength = adapt_mutation_after_breakthrough(
        mutation_rate=0.15,
        mutation_strength=0.12,
        global_best_pipes=10,
    )

    assert unchanged_rate == 0.15
    assert unchanged_strength == 0.12
    assert reduced_rate == 0.05
    assert reduced_strength == 0.03


def test_patience_logic_increases_diversity_after_no_improvement():
    (
        normal_rate,
        normal_strength,
        normal_random_ratio,
        normal_champion_ratio,
    ) = apply_patience_diversity(
        mutation_rate=0.15,
        mutation_strength=0.05,
        random_immigrant_ratio=0.1,
        champion_offspring_ratio=0.15,
        generations_without_improvement=14,
        validated_mean_pipes=3.0,
    )
    (
        boosted_rate,
        boosted_strength,
        boosted_random_ratio,
        boosted_champion_ratio,
    ) = apply_patience_diversity(
        mutation_rate=0.15,
        mutation_strength=0.05,
        random_immigrant_ratio=0.1,
        champion_offspring_ratio=0.15,
        generations_without_improvement=15,
        validated_mean_pipes=3.0,
    )

    assert normal_rate == 0.15
    assert normal_strength == 0.05
    assert normal_random_ratio == 0.1
    assert normal_champion_ratio == 0.15
    assert boosted_rate == 0.20
    assert boosted_strength > normal_strength
    assert boosted_random_ratio > normal_random_ratio
    assert boosted_champion_ratio == 0.10


def test_dynamic_champion_offspring_ratio_is_lower_for_weak_champions():
    assert get_champion_offspring_ratio(2.0) == 0.15
    assert get_champion_offspring_ratio(5.0) == 0.25
    assert get_champion_offspring_ratio(10.0) == 0.35


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

    for index in range(4):
        assert not np.shares_memory(next_generation[index], global_best)


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

    next_generation = genetic.create_next_generation(
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
    assert np.array_equal(next_generation[0], global_best)
    assert mutation_calls[0][0] == 0.02
    assert mutation_calls[0][1] == 0.03

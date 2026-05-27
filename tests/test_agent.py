import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent  # noqa: E402
from flappy_bird_ai.genetic import (  # noqa: E402
    adapt_mutation_after_breakthrough,
    adaptive_mutation_strength,
    calculate_champion_score,
    create_next_generation,
    is_better_result,
    target_reached,
)
from flappy_bird_ai.simulation import PIPE_REWARD, calculate_fitness  # noqa: E402


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


def test_fitness_increases_significantly_when_pipes_increase():
    no_pipe_fitness = calculate_fitness(frames_survived=300, pipes_passed=0)
    one_pipe_fitness = calculate_fitness(frames_survived=300, pipes_passed=1)

    assert one_pipe_fitness - no_pipe_fitness == PIPE_REWARD


def test_higher_pipe_count_beats_higher_survival_time():
    long_survival = calculate_fitness(frames_survived=5000, pipes_passed=1)
    more_pipes = calculate_fitness(frames_survived=100, pipes_passed=3)

    assert more_pipes > long_survival


def test_target_pipe_threshold_detects_success():
    assert target_reached({"pipes_passed": 10}, target_pipes=10)
    assert not target_reached({"pipes_passed": 9}, target_pipes=10)
    assert target_reached({"validated_mean_pipes": 10.0}, target_pipes=10)
    assert not target_reached({"validated_mean_pipes": 9.9}, target_pipes=10)


def test_global_best_comparison_prefers_pipes_then_fitness():
    current_best = {
        "validated_mean_pipes": 2.0,
        "validated_mean_fitness": 9000.0,
        "champion_score": 2.0,
    }
    more_pipes = {
        "validated_mean_pipes": 3.0,
        "validated_mean_fitness": 4000.0,
        "champion_score": 2.5,
    }
    same_pipes_more_score = {
        "validated_mean_pipes": 2.0,
        "validated_mean_fitness": 9500.0,
        "champion_score": 2.2,
    }
    fewer_pipes_more_fitness = {
        "validated_mean_pipes": 1.0,
        "validated_mean_fitness": 20000.0,
        "champion_score": 3.0,
    }

    assert is_better_result(more_pipes, current_best)
    assert is_better_result(same_pipes_more_score, current_best)
    assert not is_better_result(fewer_pipes_more_fitness, current_best)


def test_champion_comparison_prioritizes_validated_mean_pipes():
    current_best = {
        "validated_mean_pipes": 4.0,
        "validated_mean_fitness": 50000.0,
        "champion_score": 6.0,
    }
    candidate = {
        "validated_mean_pipes": 5.0,
        "validated_mean_fitness": 10000.0,
        "champion_score": 4.0,
    }

    assert is_better_result(candidate, current_best)


def test_unstable_champions_are_penalized_by_std_and_min_pipes():
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
        mutation_rate=0.0,
        global_best_genome=global_best,
        champion_offspring_ratio=0.5,
    )

    assert np.array_equal(next_generation[0], global_best)
    assert np.array_equal(next_generation[1], global_best)
    assert np.array_equal(next_generation[2], global_best)
    assert np.array_equal(next_generation[3], global_best)

    for index in range(4):
        assert not np.shares_memory(next_generation[index], global_best)

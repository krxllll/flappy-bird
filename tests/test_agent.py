import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.agent import BirdAgent  # noqa: E402
from flappy_bird_ai.genetic import target_reached  # noqa: E402
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


def test_target_pipe_threshold_detects_success():
    assert target_reached({"pipes_passed": 10}, target_pipes=10)
    assert not target_reached({"pipes_passed": 9}, target_pipes=10)

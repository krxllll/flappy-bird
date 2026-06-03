from concurrent.futures import ProcessPoolExecutor
import time
import warnings

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401

from flappy_bird_ai.agent import BirdAgent


PIPE_REWARD = 1000.0
PIPE_EXPONENT = 1.75


def suppress_gymnasium_observation_warnings():
    """Suppress known Gymnasium observation-space warnings from this env."""
    warnings.filterwarnings(
        "ignore",
        message=r".*obs returned by the.*method is not within the observation space.*",
        category=UserWarning,
    )


def calculate_fitness(
    frames_survived,
    pipes_passed,
    pipe_reward=PIPE_REWARD,
    pipe_exponent=PIPE_EXPONENT,
):
    """Calculate fitness from survival time and completed pipes."""
    return float(frames_survived + (pipes_passed**pipe_exponent) * pipe_reward)


def evaluate_chromosome(chromosome, render=False, frame_delay=0.0):
    """
    Run one game of Flappy Bird and return fitness, frames, and pipes passed.
    """
    suppress_gymnasium_observation_warnings()
    render_mode = "human" if render else None

    env = gym.make("FlappyBird-v0", render_mode=render_mode)
    agent = BirdAgent()

    observation, info = env.reset()

    frames_survived = 0
    pipes_passed = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.predict(observation, chromosome)
        observation, reward, terminated, truncated, info = env.step(action)
        frames_survived += 1
        pipes_passed = max(pipes_passed, int(info.get("score", 0)))

        if frame_delay > 0:
            time.sleep(frame_delay)

    env.close()

    return {
        "fitness": calculate_fitness(frames_survived, pipes_passed),
        "frames": frames_survived,
        "pipes_passed": pipes_passed,
    }


def evaluate_population(population, workers=1, executor=None):
    """Evaluate chromosomes sequentially or in parallel across processes."""
    chromosomes = list(population)

    if workers <= 1:
        return [evaluate_chromosome(chromosome) for chromosome in chromosomes]

    if executor is not None:
        return list(executor.map(evaluate_chromosome, chromosomes))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluate_chromosome, chromosomes))


def evaluate_chromosome_many(chromosome, episodes, workers=1, executor=None):
    """Evaluate the same chromosome for several independent episodes."""
    return evaluate_population(
        [chromosome] * episodes,
        workers=workers,
        executor=executor,
    )


def reevaluate_candidates(chromosomes, episodes, workers=1, executor=None):
    """Evaluate multiple chromosomes for several episodes and group results."""
    candidate_list = list(chromosomes)
    jobs = [
        chromosome
        for chromosome in candidate_list
        for _ in range(episodes)
    ]

    if not jobs:
        return []

    flat_results = evaluate_population(
        jobs,
        workers=workers,
        executor=executor,
    )

    return [
        flat_results[index * episodes : (index + 1) * episodes]
        for index in range(len(candidate_list))
    ]

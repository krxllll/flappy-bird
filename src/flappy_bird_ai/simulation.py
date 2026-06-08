from concurrent.futures import ProcessPoolExecutor
import random
import time
import warnings

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401
import numpy as np

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


def evaluate_chromosome(chromosome, render=False, frame_delay=0.0, seed=None):
    """
    Run one game of Flappy Bird and return fitness, frames, and pipes passed.
    """
    suppress_gymnasium_observation_warnings()
    render_mode = "human" if render else None

    env = gym.make("FlappyBird-v0", render_mode=render_mode)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    agent = BirdAgent()

    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)

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


def _evaluate_seeded_job(job):
    chromosome, seed = job
    return evaluate_chromosome(chromosome, seed=seed)


def evaluate_population(population, workers=1, executor=None):
    """Evaluate chromosomes sequentially or in parallel across processes."""
    chromosomes = list(population)

    if workers <= 1:
        return [evaluate_chromosome(chromosome) for chromosome in chromosomes]

    if executor is not None:
        return list(executor.map(evaluate_chromosome, chromosomes))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluate_chromosome, chromosomes))


def evaluate_chromosome_on_seeds(chromosome, seeds, workers=1, executor=None):
    """Evaluate one chromosome on a fixed list of seeds."""
    jobs = [(chromosome, seed) for seed in seeds]

    if workers <= 1:
        return [_evaluate_seeded_job(job) for job in jobs]

    if executor is not None:
        return list(executor.map(_evaluate_seeded_job, jobs))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate_seeded_job, jobs))


def evaluate_population_on_seed_batch(population, seeds, workers=1, executor=None):
    """Evaluate every chromosome on the same fixed seed batch."""
    candidate_list = list(population)
    jobs = [
        (chromosome, seed)
        for chromosome in candidate_list
        for seed in seeds
    ]

    if not jobs:
        return []

    if workers <= 1:
        flat_results = [_evaluate_seeded_job(job) for job in jobs]
    elif executor is not None:
        flat_results = list(executor.map(_evaluate_seeded_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            flat_results = list(executor.map(_evaluate_seeded_job, jobs))

    seed_count = len(seeds)
    return [
        flat_results[index * seed_count : (index + 1) * seed_count]
        for index in range(len(candidate_list))
    ]

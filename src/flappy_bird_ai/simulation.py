import time

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401

from flappy_bird_ai.agent import BirdAgent


PIPE_REWARD = 1000.0


def calculate_fitness(frames_survived, pipes_passed, pipe_reward=PIPE_REWARD):
    """Calculate fitness from survival time and completed pipes."""
    return float(frames_survived + (pipes_passed**2) * pipe_reward)


def evaluate_chromosome(chromosome, render=False, frame_delay=0.0):
    """
    Run one game of Flappy Bird and return fitness, frames, and pipes passed.
    """
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

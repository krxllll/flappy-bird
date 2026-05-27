import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.simulation import evaluate_chromosome  # noqa: E402


BEST_GENOME_PATH = PROJECT_ROOT / "outputs" / "best_bird_genome.npy"


def watch_best_bird():
    print("Loading the ultimate champion bird genome...")
    try:
        best_chromosome = np.load(BEST_GENOME_PATH)
    except FileNotFoundError:
        print(
            f"Error: '{BEST_GENOME_PATH}' not found! "
            "Please let the training finish first."
        )
        return

    print("Launching live demonstration. Press Ctrl+C in terminal to close.")
    result = evaluate_chromosome(best_chromosome, render=True, frame_delay=1 / 30)

    print(
        "Match over! "
        f"Pipes passed: {result['pipes_passed']} | "
        f"Frames survived: {result['frames']} | "
        f"Fitness: {result['fitness']:.1f}"
    )


if __name__ == "__main__":
    watch_best_bird()

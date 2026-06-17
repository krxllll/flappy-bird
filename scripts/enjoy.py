import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flappy_bird_ai.simulation import (  # noqa: E402
    evaluate_chromosome,
    suppress_gymnasium_observation_warnings,
)


BEST_GENOME_PATH = PROJECT_ROOT / "outputs" / "best_bird_genome.npy"
BEST_METADATA_PATH = PROJECT_ROOT / "outputs" / "best_bird_metadata.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Watch a saved Flappy Bird champion.")
    parser.add_argument(
        "--genome",
        type=Path,
        default=BEST_GENOME_PATH,
        help="Path to a saved genome .npy file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=BEST_METADATA_PATH,
        help="Path to the genome metadata JSON file.",
    )
    return parser.parse_args(argv)


def load_metadata(path):
    with path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_architecture_config(metadata_path):
    try:
        metadata = load_metadata(metadata_path)
    except FileNotFoundError:
        print(
            "Metadata not found; falling back to baseline architecture "
            "(input_mode=raw180, hidden_size=16)."
        )
        return {"input_mode": "raw180", "hidden_size": 16}

    return {
        "input_mode": metadata.get("input_mode", "raw180"),
        "hidden_size": int(metadata.get("hidden_size", 16)),
    }


def watch_best_bird(genome_path=BEST_GENOME_PATH, metadata_path=BEST_METADATA_PATH):
    suppress_gymnasium_observation_warnings()

    print("Loading the ultimate champion bird genome...")
    try:
        best_chromosome = np.load(genome_path)
    except FileNotFoundError:
        print(
            f"Error: '{genome_path}' not found! "
            "Please let the training finish first."
        )
        return

    architecture_config = load_architecture_config(metadata_path)
    print(
        "Using architecture: "
        f"input_mode={architecture_config['input_mode']} | "
        f"hidden_size={architecture_config['hidden_size']}"
    )
    print("Launching live demonstration. Press Ctrl+C in terminal to close.")
    result = evaluate_chromosome(
        best_chromosome,
        render=True,
        frame_delay=1 / 30,
        **architecture_config,
    )

    print(
        "Match over! "
        f"Pipes passed: {result['pipes_passed']} | "
        f"Frames survived: {result['frames']} | "
        f"Fitness: {result['fitness']:.1f}"
    )


if __name__ == "__main__":
    args = parse_args()
    watch_best_bird(genome_path=args.genome, metadata_path=args.metadata)

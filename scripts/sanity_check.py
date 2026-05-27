import sys
from pathlib import Path

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    print("Dependencies successfully imported!")
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    obs, info = env.reset()
    print(f"Initial environment observation data shape: {obs.shape}")
    print("System checklist complete. Ready to code!")
    env.close()


if __name__ == "__main__":
    main()


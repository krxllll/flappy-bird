# Flappy Bird Neuroevolution

This project trains a Flappy Bird agent with a genetic algorithm and a small feedforward neural network. Each bird is represented by one chromosome: a flat NumPy array containing all neural-network weights and biases.

The neural network architecture is:

- 180 inputs
- 16 hidden neurons
- 1 output
- 2913 chromosome parameters

The output is converted into a binary action: flap or do nothing.

## Neuroevolution Approach

Training starts with a random population of birds. Each chromosome is evaluated by running one Flappy Bird game. After each generation, the best chromosomes are kept, selected parents are crossed over, and random mutations are applied to create the next generation.

Fitness now explicitly rewards both survival and pipe passing:

```text
fitness = frames_survived + pipes_passed * PIPE_REWARD
```

`PIPE_REWARD` is defined in `src/flappy_bird_ai/simulation.py` and is currently `1000.0`. Survival time still matters, but passing pipes is the strongest signal.

## Pipe Target

Training uses `TARGET_PIPES` from `src/flappy_bird_ai/genetic.py`. The default target is `10` pipes.

At the end of each generation, the best bird is checked against this target. If the best bird reaches the target, training saves the genome and prints:

```text
Target reached: best bird passed X pipes
```

If `EARLY_STOP_ON_TARGET` is `True`, training stops immediately after the target is reached. Set it to `False` to keep training through all generations while still highlighting target success.

## Project Structure

```text
flappy-bird-neuroevolution/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── flappy_bird_ai/
│       ├── __init__.py
│       ├── agent.py
│       ├── simulation.py
│       └── genetic.py
├── scripts/
│   ├── sanity_check.py
│   ├── train.py
│   └── enjoy.py
├── outputs/
│   └── .gitkeep
└── tests/
    └── test_agent.py
```

## Outputs

Training writes generated files into `outputs/`:

- `outputs/best_bird_genome.npy`
- `outputs/training_history.csv`
- `outputs/fitness_progression.png`
- `outputs/pipe_progression.png`

The CSV tracks generation, best fitness, mean fitness, best pipes, mean pipes, best frames, and mean frames.

Generated `.npy`, `.png`, and `.csv` files are ignored by git.

## Setup on Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Commands

Run a quick environment check:

```powershell
python scripts/sanity_check.py
```

Run evolutionary training:

```powershell
python scripts/train.py
```

Run the saved best bird visually:

```powershell
python scripts/enjoy.py
```

Run tests:

```powershell
pytest
```

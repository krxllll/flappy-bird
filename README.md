# Flappy Bird Neuroevolution

This project trains a Flappy Bird agent with a genetic algorithm and a small feedforward neural network. Each bird is represented by one chromosome: a flat NumPy array containing all neural-network weights and biases.

The neural network architecture is unchanged:

- 180 inputs
- 16 hidden neurons
- 1 output
- 2913 chromosome parameters

The network output is converted into a binary action: flap or do nothing.

## Training Approach

Training starts with a random population of birds. Each chromosome is evaluated by running one Flappy Bird game. After each generation, the population is sorted, copied elites are preserved, selected parents are crossed over, and mutation creates the rest of the next generation.

Fitness is now pipe-first:

```text
fitness = frames_survived + (pipes_passed ** 2) * PIPE_REWARD
```

`PIPE_REWARD` is defined in `src/flappy_bird_ai/simulation.py` and is currently `1000.0`. Frames still provide a small continuous reward, but passing more pipes is the main objective.

## Hall Of Fame

Training keeps a global best genome across all generations. This protects against the best discovered bird being lost when later generations perform worse.

Global-best comparison uses:

1. More pipes passed
2. Higher fitness if pipe count is tied

The file `outputs/best_bird_genome.npy` is saved only when the validated global best improves.

## Validation And Target Pipes

Normal population evaluation stays fast with one episode per bird. After each generation, the top candidates are re-evaluated for several validation episodes. This reduces the chance of saving a one-time lucky run as the champion.

Training uses `TARGET_PIPES` from `src/flappy_bird_ai/genetic.py`. The default target is `10` pipes. If the validated global best reaches the target, training prints a confirmation message. If `EARLY_STOP_ON_TARGET` is `True`, training stops early only after validation confirms the target.

Mutation strength is adaptive. It starts at `INITIAL_MUTATION_STRENGTH` and gradually cools down toward `MIN_MUTATION_STRENGTH` as generations progress.

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

The CSV tracks:

- generation
- best_fitness
- mean_fitness
- best_pipes
- mean_pipes
- best_frames
- mean_frames
- global_best_pipes
- global_best_fitness

Generated `.npy`, `.png`, and `.csv` files are ignored by git.

## Setup On Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

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

If `pytest` is not on PATH, run it through the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

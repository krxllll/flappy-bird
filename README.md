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

Fitness is pipe-first:

```text
fitness = frames_survived + (pipes_passed ** 2) * PIPE_REWARD
```

`PIPE_REWARD` is defined in `src/flappy_bird_ai/simulation.py` and is currently `1000.0`. Frames still provide a small continuous reward, but passing more pipes is the main objective.

## Hall Of Fame Preservation

Training keeps a global best genome across all generations. This protects against losing a strong bird when later generations perform worse.

Global-best comparison uses validated results:

1. More validated pipes passed
2. Higher validated fitness if pipe count is tied

The file `outputs/best_bird_genome.npy` is saved only when validation confirms that the global best improved.

The global best is also injected into every new generation with `.copy()`, so it cannot be mutated accidentally. A configurable part of the next population is created from small mutations of the global best using `CHAMPION_OFFSPRING_RATIO`.

## Champion Validation

Normal population evaluation stays fast with one episode per bird. After each generation, the top candidates are re-evaluated for several validation episodes before they can replace the global best.

The current champion is also re-evaluated after each generation. These values are logged in `outputs/training_history.csv`:

- `champion_mean_pipes`
- `champion_max_pipes`
- `champion_min_pipes`

This makes it easier to see whether the saved champion is consistently strong or just had one lucky run.

## Adaptive Mutation

Mutation strength cools down gradually over training. After the validated global best reaches `BREAKTHROUGH_PIPES` pipes, currently `10`, mutation is tightened further:

```text
mutation_rate = min(current_mutation_rate, 0.08)
mutation_strength = min(current_mutation_strength, 0.05)
```

This keeps exploration early in training while making later improvements less destructive once a good behavior appears.

## Target Pipes

Training uses `TARGET_PIPES` from `src/flappy_bird_ai/genetic.py`. If the validated global best reaches this target, training prints a confirmation message. If `EARLY_STOP_ON_TARGET` is `True`, training stops early only after validation confirms the target.

## Project Structure

```text
flappy-bird-neuroevolution/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   `-- flappy_bird_ai/
|       |-- __init__.py
|       |-- agent.py
|       |-- simulation.py
|       `-- genetic.py
|-- scripts/
|   |-- sanity_check.py
|   |-- train.py
|   `-- enjoy.py
|-- outputs/
|   `-- .gitkeep
`-- tests/
    `-- test_agent.py
```

## Outputs

Training writes generated files into `outputs/`:

- `outputs/best_bird_genome.npy`
- `outputs/training_history.csv`
- `outputs/fitness_progression.png`
- `outputs/pipe_progression.png`

The CSV tracks generation best, mean generation performance, validated global champion performance, and champion re-evaluation pipe stats.

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

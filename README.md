# Flappy Bird Neuroevolution

This project trains a Flappy Bird agent with a genetic algorithm and a small feedforward neural network. Each bird is represented by one chromosome: a flat NumPy array containing all neural-network weights and biases.

The neural network architecture is unchanged:

- 180 inputs
- 16 hidden neurons
- 1 output
- 2913 chromosome parameters

The network output is converted into a binary action: flap or do nothing.

## Training Approach

Normal population evaluation stays fast: each chromosome plays one episode. Generation best and mean metrics are kept as raw per-generation telemetry.

Fitness is pipe-first:

```text
fitness = frames_survived + (pipes_passed ** PIPE_EXPONENT) * PIPE_REWARD
```

`PIPE_REWARD` and `PIPE_EXPONENT` are defined in `src/flappy_bird_ai/simulation.py`. The default exponent is `1.5`, which keeps pipe passing important while reducing extreme single-run fitness spikes.

## Validation-Based Champion Selection

The saved champion is never chosen from a single lucky generation run. Top candidates that can match or beat the current champion are re-evaluated with `VALIDATION_EPISODES = 20`.

Validated champion stats include:

- `validated_mean_pipes`
- `validated_min_pipes`
- `validated_max_pipes`
- `validated_std_pipes`
- `validated_mean_fitness`
- `champion_score`

The champion score rewards consistency:

```text
champion_score = validated_mean_pipes + 0.3 * validated_min_pipes - 0.1 * validated_std_pipes
```

Global champion comparison prioritizes validated mean pipes first, then uses this stability-aware score as a tie-breaker, then validated mean fitness as the final tie-breaker. `outputs/best_bird_genome.npy` is saved only when validation confirms improvement.

## Population Stability

Every new generation includes:

- The validated global best copied unchanged.
- About 35% conservative small mutations of the global best.
- Most remaining birds from crossover among top candidates.
- About 10% random new chromosomes for diversity.

Elites and the global best are copied with `.copy()` so they are not mutated accidentally.

Champion offspring use smaller mutation settings than normal offspring:

```text
CHAMPION_MUTATION_RATE = 0.03
CHAMPION_MUTATION_STRENGTH = 0.02
```

## Adaptive Mutation

Mutation strength cools down over training. After the validated global best reaches `BREAKTHROUGH_PIPES = 10`, mutation is tightened further:

```text
mutation_rate = min(current_mutation_rate, 0.05)
mutation_strength = min(current_mutation_strength, 0.03)
```

This keeps exploration early and makes late-stage improvements less destructive.

If the champion does not improve for several generations, patience-based diversity slightly increases the random-agent ratio and normal mutation strength. When the champion improves, the patience counter resets and normal adaptive settings resume.

## Parallel Training

Flappy Bird environment simulation is CPU-bound. GPU acceleration is not useful for this project unless the environment is rewritten for batched GPU simulation. The practical speedup is multiprocessing across CPU cores.

Population evaluation can run in parallel:

```powershell
python scripts/train.py --workers 8
python scripts/train.py --workers 16 --generations 10
```

The training script reuses one process pool across the whole run, so worker processes are not recreated for every generation. Each generation prints timing for population evaluation, top-candidate revalidation, champion validation, next-generation creation, and total generation time.

To force single-process evaluation:

```powershell
python scripts/train.py --workers 1
python scripts/train.py --no-parallel
```

## Outputs

Training writes generated files into `outputs/`:

- `outputs/best_bird_genome.npy`
- `outputs/training_history.csv`
- `outputs/fitness_progression.png`
- `outputs/pipe_progression.png`

The CSV includes raw generation metrics, revalidated generation best fitness, validated champion metrics, champion score, patience count, mutation rate, mutation strength, champion offspring ratio, and random-agent ratio. Plots include raw best/mean lines plus 5-generation moving averages, with raw generation best fitness shown as a secondary spike-prone metric.

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

# Flappy Bird Neuroevolution

This project trains a Flappy Bird agent with a genetic algorithm (GA). A chromosome is a flat NumPy array containing all weights and biases of a small neural network. See [the GA flowchart](docs/ga_flowchart.md) for the training sequence.

## Agent architecture

The environment provides 180 observation values. The agent can use them directly or average adjacent groups before passing them to a network with one configurable ReLU hidden layer and one sigmoid output. An output above 0.5 means flap; otherwise the bird does nothing.

| Input mode | Processed inputs | Preprocessing | Example hidden size | Chromosome length |
| --- | ---: | --- | ---: | ---: |
| `raw180` (default) | 180 | None | 16 | 2913 |
| `binned30` | 30 | Mean of each group of 6 values | 8 | 257 |
| `binned18` | 18 | Mean of each group of 10 values | 8 | 161 |

The chromosome length is `processed_inputs * hidden_size + hidden_size + hidden_size + 1`. The examples show one hidden size for each input mode; `--hidden-size` can be any positive integer.

## Training and validation

By default, training starts with 50 random chromosomes and runs for at most 100 generations. Each chromosome plays one episode during population evaluation. Candidates are ranked by pipes passed, then fitness, then frames survived:

```text
fitness = frames_survived + (pipes_passed ** PIPE_EXPONENT) * PIPE_REWARD
```

`PIPE_REWARD` is 1000 and `PIPE_EXPONENT` is 1.75 in `src/flappy_bird_ai/simulation.py`.

The top five candidates in each generation are validated on the same 20 seeds (`10000` through `10019`). A candidate replaces the saved global best when it improves validated mean pipes, then minimum pipes, then mean fitness. The saved champion is re-evaluated each generation. Training stops early when its validated mean pipes reaches the target (default: 10), unless `--no-early-stop` is used. After training, the champion is also evaluated on 50 separate test seeds (`20000` through `20049`).

Each next generation keeps the global best and up to five elites unchanged. It adds small mutations of the champion for 15% of the population, fills most remaining slots with crossover and mutation from the top 40% parent pool, and reserves 10% for random immigrants.

Normal offspring have a default mutation rate of 0.15. `--mutation-rate` overrides it with a value strictly between 0 and 1. Normal mutation strength decreases from 0.2 toward a minimum of 0.05 over the configured generations. Champion offspring use separate, fixed mutation settings: rate 0.03 and strength 0.02.

## Setup and commands

On Windows PowerShell, create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate the environment again.

Run a quick environment check:

```powershell
python scripts/sanity_check.py
```

Train with the defaults, or choose an architecture and mutation rate:

```powershell
python scripts/train.py
python scripts/train.py --input-mode binned18 --hidden-size 8 --mutation-rate 0.19 --workers 8 --generations 50
```

The default worker count is one fewer than the available CPU count, with a minimum of one. Use `--workers 1` or `--no-parallel` for sequential evaluation. Use `--target-pipes N` to change the early-stop target. Run `python scripts/train.py --help` for all options.

Replay the saved champion:

```powershell
python scripts/enjoy.py
```

Playback reads the saved metadata to reconstruct the agent architecture. Keep the genome `.npy` file and its metadata `.json` file together; `--genome` and `--metadata` can select another pair.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Generated outputs

Training writes these files under `outputs/`:

- `best_bird_genome.npy`: validated global best chromosome.
- `best_bird_metadata.json`: architecture, mutation settings at champion selection, validation metrics, and held-out test metrics.
- `training_history.csv`: per-generation raw fitness, pipe and frame metrics; validated global-best metrics; mutation settings; population ratios; and timing.
- `fitness_progression.png` and `pipe_progression.png`: raw generation best and mean alongside validated global-best trends.

These are run artifacts. Results depend on the chosen architecture, mutation rate, generation count, and random episodes.

Git ignores the generated `.npy`, `.png`, and `.csv` files in `outputs/`; inspect the metadata JSON before adding it to a commit.

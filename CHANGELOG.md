# Changelog

## v0.6.0

Training balance and exploration recovery update.

- Changed pipe-reward shaping to use configurable `PIPE_EXPONENT` with a default of `1.75`.
- Updated population ranking to sort candidates by pipes passed first, then fitness, then frames survived.
- Moved `sort_population_indices` into `src/flappy_bird_ai/genetic.py` so tests and training use reusable GA logic instead of importing from `scripts`.
- Increased the validation candidate pool from 5 to 10 top candidates.
- Added 3-episode revalidation for the top 10 generation candidates before logging generation-best metrics.
- Kept raw generation best fitness as a secondary spike-prone metric while logging revalidated generation best fitness separately.
- Increased champion validation and champion re-evaluation to 20 episodes.
- Preserved pipe-first champion comparison using validated mean pipes, champion score, then validated mean fitness.
- Kept champion score as a mild stability tie-breaker using mean pipes, minimum pipes, and pipe standard deviation.
- Added dynamic champion offspring ratios: lower ratios for weak champions and higher ratios after stronger validated pipe performance.
- Added conservative champion-offspring mutation settings separate from normal offspring mutation.
- Added patience-based exploration when champion improvement stalls, increasing mutation rate, mutation strength, and random immigrants while reducing champion offspring for weak champions.
- Extended CSV logging with pipe exponent, raw and revalidated generation best fitness, patience count, champion offspring ratio, and random immigrant ratio.
- Updated plots so raw generation best fitness is visually secondary and moving-average/revalidated metrics remain prominent.
- Expanded tests for pipe-first sorting, pipe exponent, champion comparison priority, dynamic champion offspring ratios, patience diversity, and safe champion offspring mutation.

## v0.5.0

Training stability and validated champion selection update.

- Added validation-based champion selection so new global best genomes are not saved from a single lucky run.
- Increased candidate validation to 10 episodes.
- Added validated champion metrics: mean pipes, minimum pipes, maximum pipes, pipe standard deviation, and mean fitness.
- Added stability-aware champion scoring using mean pipes, minimum pipes, and pipe standard deviation.
- Updated global champion comparison to prioritize validated pipe performance.
- Reduced mutation rate and mutation strength after a validated 10-pipe breakthrough.
- Increased champion-based offspring generation to stabilize the population around strong agents.
- Added random immigrant chromosomes to preserve diversity.
- Added moving-average lines to the fitness and pipe progression plots.
- Extended `training_history.csv` with validation metrics, champion score, mutation rate, and mutation strength.
- Updated tests for validated champion comparison, stability scoring, breakthrough mutation reduction, and global best preservation.

## v0.4.0

Hall-of-fame preservation and champion stability update.

- Added global best injection into each new generation.
- Added champion re-evaluation after every generation.
- Logged champion mean pipes, maximum pipes, and minimum pipes.
- Added champion-based offspring generation.
- Added adaptive mutation reduction after reaching a pipe breakthrough.
- Increased default training length.
- Improved plots to distinguish generation best results from validated champion results.
- Added tests for global best injection, champion offspring safety, mutation reduction, and elitism copying.

## v0.3.0

Validated global best and pipe-threshold training update.

- Added hall-of-fame style global best tracking.
- Preserved the best genome across generations.
- Saved `outputs/best_bird_genome.npy` only when the validated global best improved.
- Added multi-episode validation for top candidates.
- Separated generation best metrics from global champion metrics.
- Added target pipe threshold support.
- Added early stopping only after validation confirmed the pipe target.
- Added training history output to `outputs/training_history.csv`.
- Added pipe progression plotting.
- Added tests for global best comparison, target detection, adaptive mutation, and elite copying.

## v0.2.0

Pipe-aware evaluation and training metrics update.

- Added pipe count tracking during chromosome evaluation.
- Added frame survival tracking during chromosome evaluation.
- Changed evaluation to return structured metrics: fitness, frames survived, and pipes passed.
- Added a named `PIPE_REWARD` constant.
- Updated fitness calculation to reward pipe passing more strongly than survival alone.
- Updated training logs to show best fitness, mean fitness, best pipes, mean pipes, and best frames.
- Updated visual playback to print final pipes passed, frames survived, and fitness.
- Added tests for structured evaluation results and pipe-based fitness behavior.

## v0.1.0

Project restructuring and baseline neuroevolution setup.

- Reorganized the project into a clean Python structure.
- Moved reusable logic into `src/flappy_bird_ai/`.
- Moved runnable scripts into `scripts/`.
- Added `outputs/` for generated artifacts.
- Added `tests/` for automated checks.
- Added `requirements.txt` with project dependencies.
- Added `.gitignore` rules for `.venv`, cache files, generated genomes, plots, and CSV logs.
- Added a sanity check script for the Flappy Bird Gymnasium environment.
- Added baseline tests for chromosome length, chromosome-to-weight mapping, and binary action prediction.

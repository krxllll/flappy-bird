# Changelog

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


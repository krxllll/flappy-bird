# Flappy Bird Neuroevolution Project Report

## 1. Goal and project structure

This project trains a Flappy Bird controller with a genetic algorithm rather than gradient descent. Each chromosome is a flat array containing every weight and bias of a small feedforward neural network. The algorithm evaluates birds in the Gymnasium Flappy Bird environment, selects stronger chromosomes, and creates new candidates through crossover and mutation.

The project separates reusable code in `src/flappy_bird_ai/`, runnable programs in `scripts/`, tests in `tests/`, and generated training artifacts in `outputs/`. This replaced the original, less structured prototype and makes the training logic easier to inspect and test.

## 2. Agent architecture

The environment supplies 180 observation values. The controller can use all 180 values (`raw180`), average adjacent groups of 6 into 30 values (`binned30`), or average groups of 10 into 18 values (`binned18`). The hidden layer size is configurable; it uses ReLU, followed by one sigmoid output. A score above 0.5 selects flap, and any other score selects no flap.

The chromosome length follows:

```text
processed_inputs * hidden_size + hidden_size + hidden_size + 1
```

The default `raw180` mode with 16 hidden units needs 2913 parameters. With 8 hidden units, `binned30` needs 257 and `binned18` needs 161. Training writes the selected architecture to genome metadata, and visual playback reads it so the saved chromosome is mapped to the correct network.

## 3. Evaluation and fitness

Each population member plays one episode during a generation. Evaluation records frames survived, pipes passed, and fitness. The current fitness function is:

```text
fitness = frames_survived + (pipes_passed ** 1.75) * 1000
```

The pipe exponent and reward are named constants in `src/flappy_bird_ai/simulation.py`. Population ranking prioritizes pipes passed, then fitness, then frames survived. This prevents a long survival run with fewer pipes from outranking a bird that made more progress through the course.

## 4. Validation and champion selection

The top five candidates from each generation are evaluated on the same 20 fixed validation seeds, 10000 through 10019. The results include mean, minimum, maximum, and standard deviation of pipes passed, plus mean fitness. A candidate becomes the global best when it improves validated mean pipes, then validated minimum pipes, then validated mean fitness.

The saved champion is re-evaluated on the validation seeds each generation. The global best is retained across generations, independent of a single generation's raw best result. Early stopping uses the global best's validated mean pipe count against a target of 10 by default. The target can be changed, and early stopping can be disabled.

After training, the saved champion is evaluated on 50 separate test seeds, 20000 through 20049. These held-out metrics are stored separately from validation metrics in the metadata file.

## 5. Population renewal and mutation

Each new generation begins with an unchanged copy of the validated global best and up to five unchanged elites. Approximately 15% of the population consists of small mutations of the champion. Most remaining places are filled by uniform crossover between parents from the top 40% of the ranked population, followed by normal Gaussian mutation. About 10% are fresh random chromosomes.

Normal offspring use a mutation rate of 0.15 by default. The `--mutation-rate` option accepts a rate strictly between 0 and 1. Normal mutation strength decreases from 0.2 toward a floor of 0.05 over the configured number of generations. Champion offspring use separate fixed settings: mutation rate 0.03 and strength 0.02. This distinction limits changes to copies of the known champion while allowing normal offspring to explore more broadly.

## 6. Parallel evaluation and stopping

The training script can evaluate population members and validation runs across worker processes. By default it chooses one fewer worker than the available CPU count, with a minimum of one; `--workers` or `--no-parallel` changes this. A process pool is reused across generations.

The default training limit is 100 generations. Training can stop earlier when the validated global-best mean reaches the target. `--generations`, `--target-pipes`, and `--no-early-stop` control these conditions.

## 7. Outputs and interpretation

Training produces:

- `outputs/best_bird_genome.npy`: the validated global-best chromosome.
- `outputs/best_bird_metadata.json`: architecture, mutation settings at champion selection, validation metrics, and held-out test metrics.
- `outputs/training_history.csv`: raw generation best and mean metrics, validated global-best metrics, mutation settings, population ratios, worker count, and timing.
- `outputs/fitness_progression.png` and `outputs/pipe_progression.png`: simple curves for raw generation performance and validated global-best performance.

The metadata and genome form a pair for replay with `scripts/enjoy.py`. The CSV and plots show how a particular run changed over time; validation and held-out test metrics should be compared separately. A high maximum pipe count alone does not establish consistent performance, especially when the minimum remains low.

This report does not assign a single current pipe score to the project. Run results depend on the chosen architecture, mutation rate, number of generations, and game episodes. Generated outputs are not a versioned benchmark in the repository.

## 8. Testing

The automated tests cover chromosome length and weight mapping, observation aggregation, binary predictions, seeded evaluation, fitness and pipe-first ranking, champion comparison, early stopping, population preservation, mutation settings, training history, metadata, playback configuration, and plot contents. They use controlled inputs and mocked game behavior where appropriate; passing tests verifies the implementation paths, not the performance of a newly trained bird.

## 9. Conclusion

The current system combines configurable observation preprocessing and network size with a validation-based genetic algorithm. It preserves a global champion, records the settings and measurements needed to inspect a run, and can replay saved genomes with the correct architecture. Actual game performance should be reported from a named training run and its separate validation and held-out test results.

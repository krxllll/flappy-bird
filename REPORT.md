# Flappy Bird Neuroevolution Project Report

## 1. Introduction

This project trains an autonomous Flappy Bird agent using neuroevolution. The agent is controlled by a small feedforward neural network, but the network is not trained using gradient descent or backpropagation. Instead, all network weights and biases are optimized by a Genetic Algorithm.

In this approach, each possible agent is represented as a chromosome. The chromosome stores the complete neural network parameters as a flat vector of numbers. A population of these chromosomes is evaluated in the Flappy Bird environment, and better-performing chromosomes are selected, crossed over, mutated, and carried forward into later generations.

## 2. Initial Project State

The initial project state was functional but not organized as a clean Python project. Source files were not separated clearly from environment files, and some project files were placed incorrectly, possibly inside the `.venv` virtual environment directory.

There was no clean separation between reusable project logic, executable scripts, generated outputs, and tests. Files such as `agent.py`, `simulation.py`, `train.py`, `enjoy.py`, and `sanity_check.py` existed, but they were not arranged in a maintainable package structure.

The training process could run, but it was limited and not well documented. The original implementation focused mainly on survival time and saved the best genome in a simple way, which made later training improvements harder to reason about and verify.

## 3. Project Restructuring

The project was reorganized into a clean Python project layout. Source code was moved out of `.venv`, and reusable modules were placed under `src/flappy_bird_ai/`. Executable entry-point scripts were placed under `scripts/`, generated files were placed under `outputs/`, and tests were placed under `tests/`.

The `.gitignore` file was also adjusted to avoid committing virtual environments, Python cache files, pytest cache files, generated genomes, plots, and CSV logs.

The final structure is conceptually organized as follows:

```text
README.md
requirements.txt
.gitignore
src/flappy_bird_ai/agent.py
src/flappy_bird_ai/simulation.py
src/flappy_bird_ai/genetic.py
scripts/sanity_check.py
scripts/train.py
scripts/enjoy.py
outputs/
tests/
```

This structure separates project responsibilities clearly. The `src/` directory contains reusable implementation logic, `scripts/` contains runnable programs, `outputs/` stores generated training artifacts, and `tests/` contains automated checks.

## 4. Neural Network Agent

The Flappy Bird agent uses a small feedforward neural network. The input is a 180-dimensional observation vector from the Flappy Bird environment. This vector is passed through one hidden layer with 16 neurons using a ReLU activation function. The network then produces a single sigmoid output.

The sigmoid output is converted into a binary action:

- `1`: flap
- `0`: do nothing

All weights and biases are stored in a flat one-dimensional chromosome. The chromosome is mapped back into weight matrices and bias vectors whenever the agent makes a prediction.

The chromosome length is calculated as follows:

- `180 x 16 = 2880` input-to-hidden weights
- `16` hidden biases
- `16` hidden-to-output weights
- `1` output bias
- total: `2880 + 16 + 16 + 1 = 2913` parameters

The neural network architecture has remained unchanged throughout the project improvements.

## 5. Initial Genetic Algorithm

The initial training approach used a standard Genetic Algorithm. A population of candidate chromosomes was created randomly. Each chromosome represented one bird agent and was evaluated by running the Flappy Bird environment.

After evaluation, better candidates were selected. Crossover combined values from two parent chromosomes to create a child chromosome, while mutation introduced random variation into individual chromosome values. The best genome was saved so it could later be loaded by the visual playback script.

This approach provided a basic neuroevolution pipeline, but the original evaluation and selection logic did not fully prioritize the true objective of the game: passing pipes reliably.

## 6. Pipe-Based Evaluation Improvements

The evaluation logic was improved from a mostly survival-based approach into a pipe-aware evaluation system. Evaluation now tracks:

- frames survived
- pipes passed
- final fitness score

The evaluation function returns structured metrics such as `fitness`, `frames`, and `pipes_passed`, instead of only returning a single number. This makes the training loop easier to inspect, log, validate, and test.

The fitness function gives a continuous reward for survival while strongly rewarding pipe passing. This change was important because survival time alone can reward passive behavior, such as staying alive briefly without actually making progress through pipes. Passing pipes is the real objective of Flappy Bird, so pipe-based reward gives the Genetic Algorithm better learning pressure.

## 7. Pipe Threshold and Early Stopping

A configurable pipe target was added to the training process. The training loop can detect when an agent reaches a required number of pipes.

Early stopping can be enabled, but the stopping condition was improved so that training does not stop after a single lucky run. Instead, early stopping is tied to validated champion performance. This makes the target logic more reliable and prevents a one-off high score from ending training prematurely.

## 8. Global Best and Hall-of-Fame Tracking

Global best tracking was added to preserve the best genome discovered across all generations. This is separate from generation best performance.

This distinction is important because the best bird in a particular generation may perform well due to randomness, while later generations may perform worse. Without hall-of-fame tracking, a strong discovered strategy could be lost. The global best mechanism prevents this by saving and preserving the strongest validated champion.

The global best can also be injected into future generations. This helps stabilize training by ensuring that the best known strategy remains present in the population. Candidate champions are compared primarily by pipe performance and then by fitness or stability-aware scoring.

## 9. Champion Validation

Champion validation was added to reduce the effect of lucky single runs. Strong candidates are evaluated over multiple episodes before they are allowed to replace the saved global champion.

Validated champion metrics include:

- validated mean pipes
- validated minimum pipes
- validated maximum pipes
- validated pipe standard deviation
- validated mean fitness

This separates raw generation performance from reliable champion performance. The saved champion is based on validation, not only on one high-scoring generation result.

## 10. Stability-Aware Champion Selection

Champion scoring was improved to prefer consistent agents rather than agents that occasionally achieve a high score. The scoring formula considers mean pipes, minimum pipes, and the standard deviation of pipe counts.

The stability-aware score is conceptually:

```text
champion_score = validated_mean_pipes + 0.3 * validated_min_pipes - 0.1 * validated_std_pipes
```

This rewards agents that pass many pipes on average, gives extra credit to agents with better worst-case performance, and penalizes agents whose results vary too much. The goal is to produce a bird that performs reliably, not only occasionally.

## 11. Adaptive Mutation and Champion-Based Offspring

Mutation behavior was improved to support more stable training. Mutation rate and mutation strength can be reduced after a breakthrough, such as when the validated global champion reaches a significant pipe count.

This helps avoid destroying strong behavior after the agent learns a useful strategy. Early training still allows more exploration, while later training becomes more conservative around successful behaviors.

Champion-based offspring were also added. A configurable part of the next population can be generated as small mutations of the validated champion. This stabilizes the population around known successful strategies while still preserving diversity through crossover and random new chromosomes.

## 12. Logging and Visualization

Training now produces several output artifacts:

- `outputs/training_history.csv`
- `outputs/fitness_progression.png`
- `outputs/pipe_progression.png`

The training history tracks metrics such as:

- generation best fitness
- mean fitness
- validated champion fitness
- generation best pipes
- mean pipes
- validated champion mean pipes
- validated champion minimum pipes
- moving averages for better trend visualization
- mutation parameters where applicable

The plots help distinguish raw generation spikes from validated champion performance. Moving averages make long-term trends easier to interpret.

## 13. Current Results

The latest observed results show a significant improvement over the initial version:

- generation best reached approximately 30 pipes
- validated champion mean pipes reached approximately 7.6
- validated champion max pipes reached approximately 26
- mean population pipes reached approximately 4.66
- results are significantly better than the initial 3-5 pipe range

However, the model is still not perfectly stable. Validated minimum pipes can remain low, which means the champion can still fail early in some episodes. This indicates that the algorithm can discover strong agents, but consistent performance remains more difficult than occasional high performance.

The current interpretation is that the algorithm now discovers much stronger strategies, population-level performance has improved, and validation prevents overestimating lucky runs. Further stabilization would likely require stronger consistency pressure, more validation episodes, additional selection refinements, or longer training.

## 14. Testing

Tests were added and updated to verify important parts of the project. The test suite checks:

- chromosome length
- prediction output format
- fitness calculation
- pipe-based comparison
- global best logic
- champion validation logic
- adaptive mutation behavior
- safe elitism and copying behavior

These tests help ensure that the neural network architecture remains correct, that the Genetic Algorithm preserves strong agents safely, and that training stability mechanisms behave as intended.

## 15. Conclusion

The project evolved from a basic, unstructured neuroevolution prototype into a cleaner and more robust training system.

The current implementation includes:

- proper project structure
- pipe-aware fitness
- global champion preservation
- validated champion selection
- stability-aware scoring
- adaptive mutation
- champion-based offspring generation
- improved logging and plots
- better testing support

The agent architecture remains simple and unchanged, but the training pipeline is now more reliable, easier to inspect, and better suited for a university project submission.


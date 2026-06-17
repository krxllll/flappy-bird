import numpy as np


class BirdAgent:
    """Small neural-network controller encoded as a flat chromosome."""

    INPUT_SIZE = 180
    OUTPUT_SIZE = 1
    INPUT_MODE_SIZES = {
        "raw180": 180,
        "binned30": 30,
        "binned18": 18,
    }

    def __init__(self, input_mode="raw180", hidden_size=16):
        if input_mode not in self.INPUT_MODE_SIZES:
            supported = ", ".join(sorted(self.INPUT_MODE_SIZES))
            raise ValueError(
                f"Unsupported input_mode '{input_mode}'. Expected one of: {supported}."
            )
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")

        self.input_mode = input_mode
        self.input_size = self.INPUT_SIZE
        self.processed_input_size = self.INPUT_MODE_SIZES[input_mode]
        self.hidden_size = hidden_size
        self.output_size = 1

        self.chromosome_length = (
            self.processed_input_size * self.hidden_size + self.hidden_size
        ) + (self.hidden_size * self.output_size + self.output_size)

    def preprocess_observation(self, observation):
        """Convert the raw 180-value observation into the configured feature vector."""
        values = np.asarray(observation, dtype=float).reshape(-1)
        if len(values) != self.input_size:
            raise ValueError(
                f"Expected observation with {self.input_size} values, got {len(values)}."
            )

        if self.input_mode == "raw180":
            return values

        if self.input_mode == "binned30":
            group_size = 6
        elif self.input_mode == "binned18":
            group_size = 10
        else:
            raise ValueError(f"Unsupported input_mode '{self.input_mode}'.")

        # Mean aggregation reduces dimensionality while smoothing local observation noise.
        return values.reshape(-1, group_size).mean(axis=1)

    def map_chromosome_to_weights(self, chromosome):
        """
        Convert a flat 1D chromosome into neural-network weights and biases.
        """
        chromosome = np.asarray(chromosome, dtype=float)
        if len(chromosome) != self.chromosome_length:
            raise ValueError(
                "Chromosome length mismatch for "
                f"input_mode={self.input_mode}, hidden_size={self.hidden_size}: "
                f"expected {self.chromosome_length}, got {len(chromosome)}."
            )

        idx = 0

        w1_end = idx + (self.processed_input_size * self.hidden_size)
        W1 = chromosome[idx:w1_end].reshape(
            self.processed_input_size,
            self.hidden_size,
        )
        idx = w1_end

        b1_end = idx + self.hidden_size
        b1 = chromosome[idx:b1_end].reshape(1, self.hidden_size)
        idx = b1_end

        w2_end = idx + (self.hidden_size * self.output_size)
        W2 = chromosome[idx:w2_end].reshape(self.hidden_size, self.output_size)
        idx = w2_end

        b2_end = idx + self.output_size
        b2 = chromosome[idx:b2_end].reshape(1, self.output_size)

        return W1, b1, W2, b2

    def predict(self, observation, chromosome):
        """
        Run a forward pass and return 0 for no flap or 1 for flap.
        """
        W1, b1, W2, b2 = self.map_chromosome_to_weights(chromosome)

        X = self.preprocess_observation(observation).reshape(1, -1)

        hidden_activation = np.dot(X, W1) + b1
        hidden_output = np.maximum(0, hidden_activation)

        output_activation = np.dot(hidden_output, W2) + b2
        output_score = 1 / (1 + np.exp(-output_activation))

        return 1 if output_score[0][0] > 0.5 else 0


if __name__ == "__main__":
    agent = BirdAgent()
    print("Agent brain initialized!")
    print(
        "Each bird strategy requires a chromosome string of exactly "
        f"{agent.chromosome_length} numbers."
    )

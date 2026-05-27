import numpy as np


class BirdAgent:
    """Small neural-network controller encoded as a flat chromosome."""

    def __init__(self):
        self.input_size = 180
        self.hidden_size = 16
        self.output_size = 1

        # Hidden layer: (180 * 16) weights + 16 biases = 2896
        # Output layer: (16 * 1) weights + 1 bias = 17
        # Total = 2913 parameters
        self.chromosome_length = (
            self.input_size * self.hidden_size + self.hidden_size
        ) + (self.hidden_size * self.output_size + self.output_size)

    def map_chromosome_to_weights(self, chromosome):
        """
        Convert a flat 1D chromosome into neural-network weights and biases.
        """
        idx = 0

        w1_end = idx + (self.input_size * self.hidden_size)
        W1 = chromosome[idx:w1_end].reshape(self.input_size, self.hidden_size)
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

        X = observation.reshape(1, -1)

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


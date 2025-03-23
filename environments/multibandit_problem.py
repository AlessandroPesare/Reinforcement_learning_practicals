import numpy as np

class BanditEnvironment:
    def __init__(self, k=10):
        """
        Initializes the multi-armed bandit environment.

        Parameters:
            k (int): Number of arms/actions available (default: 10).
        """
        self.k = k
        self.reset()  # Initialize true values and optimal action

    def reset(self):
        """
        Resets the environment by:
        1. Generating new true reward values for each arm
        2. Identifying the optimal arm (ground truth)
        """
        # True expected rewards for each arm ~ Normal(μ=0, σ=1)
        self.true_values = np.random.normal(0, 1, self.k)

        # Index of the arm with highest true reward (optimal action)
        self.optimal_action = np.argmax(self.true_values)

    def get_reward(self, action):
        """
        Simulates pulling a bandit arm and receiving a stochastic reward.

        Parameters:
            action (int): Index of the chosen arm (0 to k-1).

        Returns:
            float: Sampled reward from Normal(μ=true_value_of_arm, σ=1)
        """
        # Reward = True value + Gaussian noise (σ=1 for exploration challenge)
        reward = np.random.normal(self.true_values[action], 1)
        return reward
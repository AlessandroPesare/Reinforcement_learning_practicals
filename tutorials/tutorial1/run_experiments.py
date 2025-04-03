import numpy as np
import matplotlib.pyplot as plt

# Import the BanditEnvironment from our custom multi-armed bandit implementation.
from environments.multibandit_problem import BanditEnvironment


def run_experiment_epsilon(env_class, epsilon, num_episodes=2000, time_steps=1000, **env_kwargs):
    """
    Runs the k-armed bandit experiment using the epsilon-greedy policy.

    Parameters:
        env_class (class): The environment class to use.
        epsilon (float): Probability of selecting a random action (exploration).
        num_episodes (int): Number of independent episodes.
        time_steps (int): Number of steps per episode.
        env_kwargs: Additional keyword arguments to pass to the environment constructor (e.g., k=10).

    Returns:
        avg_reward (np.ndarray): The average reward at each time step (averaged over episodes).
        optimal_action_percent (np.ndarray): The percentage of times the optimal action was chosen.
    """
    rewards = np.zeros((num_episodes, time_steps))
    optimal_actions = np.zeros((num_episodes, time_steps))

    for episode in range(num_episodes):
        env = env_class(**env_kwargs)
        # Initialize estimated action values (Q) and action counts.
        Q = np.zeros(env.k)
        action_counts = np.zeros(env.k)

        for t in range(time_steps):
            # Epsilon-greedy action selection:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.k)
            else:
                # In case of ties, break them randomly.
                max_estimate = np.max(Q)
                candidates = np.where(Q == max_estimate)[0]
                action = np.random.choice(candidates)

            # Check if the selected action is the optimal action.
            if action == env.optimal_action:
                optimal_actions[episode, t] = 1

            # Obtain reward from the environment.
            reward = env.get_reward(action)
            rewards[episode, t] = reward

            # Update the estimated value using the sample-average method.
            action_counts[action] += 1
            Q[action] += (reward - Q[action]) / action_counts[action]

    # Calculate the average reward and optimal action percentage over episodes.
    avg_reward = rewards.mean(axis=0)
    optimal_action_percent = optimal_actions.mean(axis=0) * 100
    return avg_reward, optimal_action_percent


def run_experiment_ucb(env_class, c, num_episodes=2000, time_steps=1000, **env_kwargs):
    """
    Runs the k-armed bandit experiment using the Upper Confidence Bound (UCB) policy.

    The UCB action selection formula is:
        A_t = argmax_a ( Q_t(a) + c * sqrt( ln(t+1) / N_t(a) ) ),
    where:
        Q_t(a) is the estimated value for action a,
        N_t(a) is the number of times action a has been selected,
        c is a parameter controlling exploration.

    Parameters:
        env_class (class): The environment class to use.
        c (float): Exploration parameter for UCB.
        num_episodes (int): Number of independent episodes.
        time_steps (int): Number of steps per episode.
        env_kwargs: Additional keyword arguments to pass to the environment constructor (e.g., k=10).

    Returns:
        avg_reward (np.ndarray): The average reward at each time step (averaged over episodes).
        optimal_action_percent (np.ndarray): The percentage of times the optimal action was chosen.
    """
    rewards = np.zeros((num_episodes, time_steps))
    optimal_actions = np.zeros((num_episodes, time_steps))

    for episode in range(num_episodes):
        env = env_class(**env_kwargs)
        # Initialize estimated action values (Q) and action counts.
        Q = np.zeros(env.k)
        counts = np.zeros(env.k)

        for t in range(time_steps):
            if t < env.k:
                # Ensure each action is tried at least once to avoid division by zero.
                action = t
            else:
                # Compute UCB values for each action.
                ucb_values = np.zeros(env.k)
                for a in range(env.k):
                    # If an action has not been selected, set its UCB value to infinity.
                    if counts[a] == 0:
                        ucb_values[a] = float('inf')
                    else:
                        ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / counts[a])
                action = np.argmax(ucb_values)

            # Check if the selected action is optimal.
            if action == env.optimal_action:
                optimal_actions[episode, t] = 1

            # Obtain the reward from the environment.
            reward = env.get_reward(action)
            rewards[episode, t] = reward

            # Update the action count and the estimated value for the chosen action.
            counts[action] += 1
            Q[action] += (reward - Q[action]) / counts[action]

    # Compute the average reward and the percentage of optimal actions.
    avg_reward = rewards.mean(axis=0)
    optimal_action_percent = optimal_actions.mean(axis=0) * 100
    return avg_reward, optimal_action_percent


def run_experiment_softmax(env_class, tau, num_episodes=2000, time_steps=1000, **env_kwargs):
    """
    Runs the k-armed bandit experiment using softmax action selection.

    In softmax selection, the probability of selecting an action a is given by:

        P(a) = exp(Q_t(a)/tau) / sum(exp(Q_t(i)/tau) for i in all actions)

    Parameters:
        env_class (class): The environment class to use.
        tau (float): Temperature parameter that controls exploration.
                     - Low tau makes the selection nearly greedy.
                     - High tau makes the selection nearly uniform (random).
        num_episodes (int): Number of independent episodes.
        time_steps (int): Number of steps per episode.
        env_kwargs: Additional keyword arguments for the environment constructor (e.g., k=10).

    Returns:
        avg_reward (np.ndarray): The average reward at each time step (averaged over episodes).
        optimal_action_percent (np.ndarray): The percentage of times the optimal action was chosen.
    """
    rewards = np.zeros((num_episodes, time_steps))
    optimal_actions = np.zeros((num_episodes, time_steps))

    for episode in range(num_episodes):
        env = env_class(**env_kwargs)
        # Initialize estimated action values (Q) and the count for each action.
        Q = np.zeros(env.k)
        action_counts = np.zeros(env.k)

        for t in range(time_steps):
            # Compute the softmax probabilities for each action.
            # We use Q/tau to scale the action values.
            exp_values = np.exp(Q / tau)
            probs = exp_values / np.sum(exp_values)

            # Select an action according to the probability distribution.
            action = np.random.choice(np.arange(env.k), p=probs)

            # Check if the selected action is optimal.
            if action == env.optimal_action:
                optimal_actions[episode, t] = 1

            # Get the reward for the chosen action.
            reward = env.get_reward(action)
            rewards[episode, t] = reward

            # Update the count and the estimated value for the action using sample averaging.
            action_counts[action] += 1
            Q[action] += (reward - Q[action]) / action_counts[action]

    # Compute the average reward and the optimal action percentage over episodes.
    avg_reward = rewards.mean(axis=0)
    optimal_action_percent = optimal_actions.mean(axis=0) * 100
    return avg_reward, optimal_action_percent


if __name__ == "__main__":
    # Set the environment class to test (BanditEnvironment for the multi-armed bandit problem).
    env_class = BanditEnvironment

    # Define experimental parameters.
    time_steps = 1000
    num_episodes = 2000
    k = 10

    # --- Epsilon-Greedy Experiment ---
    epsilons = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 1]
    epsilon_results = {}

    for eps in epsilons:
        avg_reward, optimal_action_percent = run_experiment_epsilon(env_class, eps, num_episodes, time_steps, k=k)
        epsilon_results[eps] = (avg_reward, optimal_action_percent)

    # Plot average reward for epsilon-greedy.
    plt.figure(figsize=(12, 5))
    for eps in epsilons:
        avg_reward, _ = epsilon_results[eps]
        plt.plot(avg_reward, label=f'epsilon = {eps}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('Epsilon-Greedy: Average Reward vs. Time Steps')
    plt.legend()
    plt.show()

    # Plot the percentage of optimal action selections over time for each epsilon value.
    plt.figure(figsize=(12, 5))
    for eps in epsilons:
        _, optimal_action_percent = epsilon_results[eps]
        plt.plot(optimal_action_percent, label=f'epsilon = {eps}')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Optimal Action Percentage vs. Time Steps per Epsilon')
    plt.legend()
    plt.show()

    # --- UCB Experiment ---
    # Define different c values to test.
    c_values = [0.5, 1, 2, 5]
    ucb_results = {}

    for c in c_values:
        avg_reward, optimal_action_percent = run_experiment_ucb(env_class, c, num_episodes, time_steps, k=k)
        ucb_results[c] = (avg_reward, optimal_action_percent)

    # Plot average reward for UCB.
    plt.figure(figsize=(12, 5))
    for c in c_values:
        avg_reward, _ = ucb_results[c]
        plt.plot(avg_reward, label=f'c = {c}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('UCB: Average Reward vs. Time Steps')
    plt.legend()
    plt.show()

    # Plot the percentage of optimal actions for UCB.
    plt.figure(figsize=(12, 5))
    for c in c_values:
        _, optimal_action_percent = ucb_results[c]
        plt.plot(optimal_action_percent, label=f'c = {c}')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action')
    plt.title('UCB: Optimal Action Percentage vs. Time Steps')
    plt.legend()
    plt.show()

    # Define different temperature (tau) values to test.
    tau_values = [0.01, 0.1, 1, 10]
    softmax_results = {}

    for tau in tau_values:
        avg_reward, optimal_action_percent = run_experiment_softmax(env_class, tau, num_episodes, time_steps, k=k)
        softmax_results[tau] = (avg_reward, optimal_action_percent)

    # Plot average reward for different temperature values.
    plt.figure(figsize=(12, 5))
    for tau in tau_values:
        avg_reward, _ = softmax_results[tau]
        plt.plot(avg_reward, label=f'tau = {tau}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('Softmax: Average Reward vs. Time Steps')
    plt.legend()
    plt.show()

    # Plot percentage of optimal actions for different temperature values.
    plt.figure(figsize=(12, 5))
    for tau in tau_values:
        _, optimal_action_percent = softmax_results[tau]
        plt.plot(optimal_action_percent, label=f'tau = {tau}')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Softmax: Optimal Action Percentage vs. Time Steps')
    plt.legend()
    plt.show()

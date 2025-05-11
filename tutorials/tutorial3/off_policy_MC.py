import sys
import os
from environments.grid_world  import SimpleGridWorld
import numpy as np
import random

def off_policy_mc_control(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Off-policy MC control using weighted importance sampling.
    Finds optimal target policy.
    Returns: Q, policy
    """
    # Initialize
    Q = {}  # action-value estimates Q[s][a]
    C = {}  # cumulative weights C[s][a]
    policy = {}  # target policy pi[s] = best action

    # Initialize Q, C, and policy for all state-action pairs
    for state in range(env.observation_space.n):
        Q[state] = np.zeros(env.action_space.n)
        C[state] = np.zeros(env.action_space.n)
        policy[state] = 0

    for i_episode in range(1, num_episodes + 1):
        # Generate an episode using behavior policy b
        episode = []  # list of (state, action, reward)
        state = env.reset()
        state_idx = state[0] * env.grid_size + state[1]
        done = False
        while not done:
            action = behavior_policy(state_idx, env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            next_state_idx = next_state[0] * env.grid_size + next_state[1]
            episode.append((state_idx, action, reward))
            state_idx = next_state_idx

        G = 0.0
        W = 1.0
        # Process episode in reverse
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = discount_factor * G + reward_t
            C[state_t][action_t] += W
            Q[state_t][action_t] += (W / C[state_t][action_t]) * (G - Q[state_t][action_t])
            # Update target policy to greedy
            best_action = np.argmax(Q[state_t])
            policy[state_t] = best_action

            if action_t != best_action:
                break
            # update importance sampling ratio
            # behavior policy is uniform random
            b_prob = 1.0 / env.action_space.n
            W = W * (1.0 / b_prob)

    return Q, policy

# Example usage
if __name__ == "__main__":
    env = SimpleGridWorld()
    # Behavior policy: uniform random
    def behavior_policy(state, n_actions):
        return np.random.choice(np.arange(n_actions))

    Q, optimal_policy = off_policy_mc_control(env, num_episodes=1000000, behavior_policy=behavior_policy)

    # Display optimal policy as arrows
    action_symbols = {0: '^', 1: 'v', 2: '<', 3: '>'}
    grid_policy = np.full((env.grid_size, env.grid_size), ' ')
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            state = r * env.grid_size + c
            if (r, c) in env.terminal_states:
                grid_policy[r, c] = 'T'
            else:
                grid_policy[r, c] = action_symbols[optimal_policy[state]]
    for row in grid_policy:
        print(' '.join(row))

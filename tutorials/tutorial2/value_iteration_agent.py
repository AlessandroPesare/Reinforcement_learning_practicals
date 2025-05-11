import numpy as np

class ValueIterationAgent:
    """
    Implements Value Iteration by updating the value function using the Bellman optimality equation,
    and then extracting the optimal policy.
    """
    def __init__(self, env, theta=1e-4, gamma=1.0):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.value = np.zeros((env.grid_size, env.grid_size))
        self.policy = np.zeros((env.grid_size, env.grid_size), dtype=int)

    def iterate_value(self):
        while True:
            delta = 0
            for r in range(self.env.grid_size):
                for c in range(self.env.grid_size):
                    state = (r, c)
                    if state in self.env.terminal_states:
                        continue
                    v_old = self.value[r, c]
                    q_values = []
                    for action in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step(state, action)
                        nr, nc = next_state
                        q_values.append(reward + self.gamma * self.value[nr, nc])
                    self.value[r, c] = max(q_values)
                    delta = max(delta, abs(v_old - self.value[r, c]))
            if delta < self.theta:
                break
        # Extract the optimal policy from the computed value function
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                state = (r, c)
                if state in self.env.terminal_states:
                    continue
                q_values = []
                for action in range(self.env.action_space.n):
                    next_state, reward, done, _ = self.env.simulate_step(state, action)
                    nr, nc = next_state
                    q_values.append(reward + self.gamma * self.value[nr, nc])
                self.policy[r, c] = np.argmax(q_values)
        return self.policy, self.value

import numpy as np
class PolicyIterationAgent:
    """
    Implements Policy Iteration by alternating policy evaluation and policy improvement
    until the optimal policy is obtained.
    """
    def __init__(self, env, theta=1e-4, gamma=0.9):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.value = np.zeros((env.grid_size, env.grid_size))
        # Initialize the policy arbitrarily (all actions set to 0 initially)
        self.policy = np.zeros((env.grid_size, env.grid_size), dtype=int)

    def policy_evaluation(self):
        while True:
            delta = 0
            for r in range(self.env.grid_size):
                for c in range(self.env.grid_size):
                    state = (r, c)
                    if state in self.env.terminal_states:
                        continue
                    v_old = self.value[r, c]
                    action = self.policy[r, c]
                    next_state, reward, done, _ = self.env.simulate_step(state, action)
                    nr, nc = next_state
                    self.value[r, c] = reward + self.gamma * self.value[nr, nc]
                    delta = max(delta, abs(v_old - self.value[r, c]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                state = (r, c)
                if state in self.env.terminal_states:
                    continue
                old_action = self.policy[r, c]
                q_values = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    next_state, reward, done, _ = self.env.simulate_step(state, action)
                    nr, nc = next_state
                    q_values[action] = reward + self.gamma * self.value[nr, nc]
                best_action = np.argmax(q_values)
                self.policy[r, c] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    def iterate_policy(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.value

import numpy as np

class PolicyEvaluator:
    """
    Performs iterative policy evaluation using an equiprobable policy
    (each action has a probability of 1/4).
    """
    def __init__(self, env, theta=1e-4, gamma=1.0):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.value = np.zeros((env.grid_size, env.grid_size))

    def evaluate(self):
        while True:
            delta = 0
            for r in range(self.env.grid_size):
                for c in range(self.env.grid_size):
                    state = (r, c)
                    if state in self.env.terminal_states:
                        continue
                    v_old = self.value[r, c]
                    v_new = 0
                    for action in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step(state, action)
                        nr, nc = next_state
                        # Equiprobable policy: each action chosen with probability 1/4
                        v_new += (1 / self.env.action_space.n) * (reward + self.gamma * self.value[nr, nc])
                    self.value[r, c] = v_new
                    delta = max(delta, abs(v_old - v_new))
            if delta < self.theta:
                break
        return self.value
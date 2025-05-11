import gym
from gym import spaces
import numpy as np

class SimpleGridWorld(gym.Env):
    """
    4x4 Grid World Environment:
      - State: represented by a pair (row, column)
      - Terminal states: (0, 0) (top-left corner) and (3, 3) (bottom-right corner)
      - Actions: 0=Up, 1=Down, 2=Left, 3=Right (deterministic movements)
      - Reward: -1 per step, 0 if a terminal state is reached
      - max_steps: maximum number of steps per episode
    """
    def __init__(self, max_steps=100):
        super(SimpleGridWorld, self).__init__()
        self.grid_size = 4  # 4x4 grid
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.terminal_states = [(0, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.start_state = (0, self.grid_size - 2)  # for example, near the right side of the top row
        self.state = self.start_state
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start_state
        self.current_step = 0
        return self.state

    def step(self, action):
        """
        Execute an action in the environment.
        If the state is terminal, return immediately.
        """
        if self.state in self.terminal_states:
            return self.state, 0, True, {}
        r, c = self.state
        if action == 0:  # UP
            r = max(0, r - 1)
        elif action == 1:  # DOWN
            r = min(self.grid_size - 1, r + 1)
        elif action == 2:  # LEFT
            c = max(0, c - 1)
        elif action == 3:  # RIGHT
            c = min(self.grid_size - 1, c + 1)
        next_state = (r, c)
        self.state = next_state
        self.current_step += 1
        reward = -1
        done = False
        if next_state in self.terminal_states:
            reward = 0
            done = True
        if self.current_step >= self.max_steps:
            done = True
        return next_state, reward, done, {}

    def simulate_step(self, state, action):
        """
        Helper method to simulate the transition from a given state and action
        without modifying the internal state of the environment.
        """
        if state in self.terminal_states:
            return state, 0, True, {}
        r, c = state
        if action == 0:  # UP
            r = max(0, r - 1)
        elif action == 1:  # DOWN
            r = min(self.grid_size - 1, r + 1)
        elif action == 2:  # LEFT
            c = max(0, c - 1)
        elif action == 3:  # RIGHT
            c = min(self.grid_size - 1, c + 1)
        next_state = (r, c)
        reward = -1
        done = next_state in self.terminal_states
        if done:
            reward = 0
        return next_state, reward, done, {}

    def render(self):
        """Display the grid with the agent and terminal states."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        # Mark terminal states
        for (r, c) in self.terminal_states:
            grid[r, c] = "T"
        # Mark the agent's current position
        r, c = self.state
        grid[r, c] = "A"
        for row in grid:
            print(" ".join(row))
        print()

    def close(self):
        pass
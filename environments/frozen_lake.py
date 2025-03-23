import gym
from gym import spaces
import numpy as np
import random


class FrozenLake(gym.Env):
    def __init__(self, size=4, slippery=False):
        super(FrozenLake, self).__init__()

        self.size = size
        self.slippery = slippery

        # Dimensione della griglia (size x size)
        self.grid_height = size
        self.grid_width = size

        # Spazio delle azioni: 4 direzioni (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Spazio degli stati: ogni stato è un (x, y) nella griglia
        self.observation_space = spaces.Discrete(self.grid_height * self.grid_width)

        # Inizializzazione della posizione del giocatore
        self.start_position = (0, 0)
        self.goal_position = (size - 1, size - 1)
        self.state = self.start_position

        # Definizione dei fori (terminal states)
        self.holes = [(1, 2), (2, 1), (3, 3)]

    def reset(self):
        """Resetta l'ambiente"""
        self.state = self.start_position
        return self.state  # Restituisce lo stato iniziale

    def step(self, action):
        """Esegui un'azione nell'ambiente"""
        x, y = self.state

        # Mappatura delle azioni (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.grid_height - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.grid_width - 1, y + 1)

        # Applicazione del comportamento scivoloso (se abilitato)
        if self.slippery:
            possible_actions = [0, 1, 2, 3]
            random.shuffle(possible_actions)
            action = possible_actions[0]
            return self.step(action)

        self.state = (x, y)

        # Verifica se l'agente è caduto nel buco
        if self.state in self.holes:
            reward = -100
            done = True
        else:
            reward = -1  # Penalità per ogni passo
            done = False

        # Verifica se siamo arrivati al goal
        if self.state == self.goal_position:
            reward = 0  # Ricompensa quando arriviamo al goal
            done = True

        return self.state, reward, done, False, {}

    def render(self):
        """Visualizza lo stato dell'ambiente"""
        grid = np.zeros((self.grid_height, self.grid_width), dtype=str)
        grid[:] = "."

        x, y = self.state
        grid[x, y] = "A"  # 'A' rappresenta l'agente
        gx, gy = self.goal_position
        grid[gx, gy] = "G"  # 'G' rappresenta l'obiettivo

        # Mostra i fori
        for (hx, hy) in self.holes:
            grid[hx, hy] = "H"  # 'H' rappresenta il buco

        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass

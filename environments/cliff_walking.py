import gym
from gym import spaces
import numpy as np


class CliffWalking(gym.Env):
    def __init__(self):
        super(CliffWalking, self).__init__()

        # Dimensione della griglia 12x4 (12 righe, 4 colonne)
        self.grid_height = 12
        self.grid_width = 4

        # Spazio delle azioni: 4 direzioni (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Spazio degli stati: ogni stato è un (x, y) nella griglia
        self.observation_space = spaces.Discrete(self.grid_height * self.grid_width)

        # Inizializzazione della posizione del giocatore (in basso a sinistra)
        self.start_position = (11, 0)
        self.goal_position = (11, 3)
        self.state = self.start_position

        # Posizione del cliff (il bordo tra start e goal)
        self.cliff = [(11, i) for i in range(1, 3)]

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

        self.state = (x, y)

        # Verifica se l'agente è caduto nel cliff
        if self.state in self.cliff:
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

        # Mostra il cliff
        for (cx, cy) in self.cliff:
            grid[cx, cy] = "C"  # 'C' rappresenta il cliff

        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass

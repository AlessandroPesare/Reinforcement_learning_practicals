import gym
from gym import spaces
import numpy as np


class WindyGridWorld(gym.Env):
    def __init__(self):
        super(WindyGridWorld, self).__init__()

        # Dimensione della griglia 10x7 (10 righe, 7 colonne)
        self.grid_height = 10
        self.grid_width = 7

        # Spazio delle azioni: 4 direzioni (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Spazio degli stati: ogni stato è un (x, y) nella griglia
        self.observation_space = spaces.Discrete(self.grid_height * self.grid_width)

        # Inizializzazione della posizione del giocatore (al centro della prima colonna)
        self.start_position = (0, 3)
        self.goal_position = (0, 6)
        self.state = self.start_position

        # Mappa del vento
        self.wind_map = [0, 0, 0, 1, 1, 2, 2, 1]  # La quantità di vento per ogni colonna

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

        # Gestione del vento (spostamento casuale verso l'alto)
        if x >= 3 and x <= 5:  # Le righe 3, 4 e 5 sono ventose
            x = max(0, x - self.wind_map[y])

        self.state = (x, y)

        # Verifica se siamo arrivati al goal
        done = self.state == self.goal_position
        reward = -1  # Penalità per ogni passo
        if done:
            reward = 0  # Ricompensa quando arriviamo al goal

        return self.state, reward, done, False, {}

    def render(self):
        """Visualizza lo stato dell'ambiente"""
        grid = np.zeros((self.grid_height, self.grid_width), dtype=str)
        grid[:] = "."

        x, y = self.state
        grid[x, y] = "A"  # 'A' rappresenta l'agente
        gx, gy = self.goal_position
        grid[gx, gy] = "G"  # 'G' rappresenta l'obiettivo

        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass

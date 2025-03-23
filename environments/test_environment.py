import numpy as np
import matplotlib.pyplot as plt

# Importiamo gli ambienti dal package environments
from environments.windy_grid_world import WindyGridWorld
from environments.cliff_walking import CliffWalking
from environments.frozen_lake import FrozenLake
from environments.blackjack import Blackjack
from environments.multibandit_problem import BanditEnvironment


def test_windy_grid_world():
    print("Testing WindyGridWorld environment...")
    env = WindyGridWorld()  # Creiamo l'istanza dell'ambiente
    state, _ = env.reset()  # Resettiamo l'ambiente per ottenere lo stato iniziale
    done = False
    while not done:
        # Azione casuale (es. UP o altra, a seconda di come è definito l'ambiente)
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    env.close()
    print("WindyGridWorld test completed.\n")


def test_cliff_walking():
    print("Testing CliffWalking environment...")
    env = CliffWalking()  # Creiamo l'istanza dell'ambiente
    state, _ = env.reset()  # Stato iniziale
    done = False
    while not done:
        action = env.action_space.sample()  # Azione casuale
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    env.close()
    print("CliffWalking test completed.\n")


def test_frozen_lake():
    print("Testing FrozenLake environment...")
    env = FrozenLake()  # Creiamo l'istanza dell'ambiente
    state, _ = env.reset()  # Stato iniziale
    done = False
    while not done:
        action = env.action_space.sample()  # Azione casuale
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    env.close()
    print("FrozenLake test completed.\n")


def test_blackjack():
    print("Testing Blackjack environment...")
    env = Blackjack()
    state = env.reset()
    print(f"Initial state: {state}")
    done = False
    while not done:
        action = env.action_space.sample()  # Azione casuale (0 = hit, 1 = stick)
        print(f"Taking action: {'hit' if action == 0 else 'stick'}")
        state, reward, done, _ = env.step(action)
        env.render()  # Visualizza lo stato corrente
        print(f"Reward: {reward}")
    print("Blackjack test completed.\n")


def test_multi_bandit():
    """
    Testa il problema del multi-armed bandit utilizzando una politica epsilon-greedy.
    Qui si assume che BanditEnvironment (definita in environments/bandit_environment.py)
    implementi un ambiente in cui:
      - Il numero di bracci è specificato da k (default=10)
      - Il metodo reset() inizializza i valori veri e determina l'azione ottimale (self.optimal_action)
      - Il metodo get_reward(action) restituisce il reward per l'azione scelta,
        campionato normalmente attorno al valore vero.
    """
    print("Testing Multi-Armed Bandit environment...")
    env = BanditEnvironment(k=10)
    total_steps = 1000
    epsilon = 0.1  # Parametro epsilon della politica epsilon-greedy
    Q = np.zeros(env.k)  # Stima dei valori per ogni braccio
    counts = np.zeros(env.k)  # Conteggio delle volte in cui ogni braccio è stato scelto
    rewards = []
    optimal_action_selections = 0

    # Identifichiamo l'azione ottimale in base ai valori veri generati in reset()
    optimal_action = env.optimal_action

    for step in range(total_steps):
        # Selezione dell'azione con epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(0, env.k)
        else:
            max_val = np.max(Q)
            candidates = np.where(Q == max_val)[0]
            action = np.random.choice(candidates)

        # Otteniamo il reward per l'azione scelta
        reward = env.get_reward(action)
        rewards.append(reward)

        # Aggiorniamo le stime utilizzando la media campionaria
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        # Contiamo se l'azione scelta corrisponde a quella ottimale
        if action == optimal_action:
            optimal_action_selections += 1

    avg_reward = np.mean(rewards)
    optimal_percentage = (optimal_action_selections / total_steps) * 100

    print(f"Average reward over {total_steps} steps: {avg_reward:.2f}")
    print(f"Percentage of optimal actions taken: {optimal_percentage:.2f}%")
    print("Multi-Armed Bandit test completed.\n")


def main():
    # Eseguiamo tutti i test degli ambienti
    test_windy_grid_world()
    test_cliff_walking()
    test_frozen_lake()
    test_blackjack()
    test_multi_bandit()


if __name__ == "__main__":
    main()

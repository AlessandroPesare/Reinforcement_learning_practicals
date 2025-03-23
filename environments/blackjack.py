import random
import gym
from gym import spaces


class Blackjack(gym.Env):
    """
    A simple Blackjack environment for reinforcement learning.
    The agent can either 'hit' (draw a card) or 'stick' (stop drawing cards).
    The goal is to get as close to 21 points without going over.
    """

    def __init__(self):
        super(Blackjack, self).__init__()

        # Definizione dello spazio delle azioni
        # 0 -> "hit" (chiedere una carta), 1 -> "stick" (fermarsi)
        self.action_space = spaces.Discrete(2)

        # Definizione dello spazio degli stati
        # Stato: (somma delle carte, carta visibile del mazziere, se c'è un asso che può essere contato come 11)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # somma delle carte, 0-31
            spaces.Discrete(11),  # carta visibile del mazziere, 1-10
            spaces.Discrete(2)  # se c'è un asso che può essere contato come 11 (0 o 1)
        ))

        # Elenco delle carte (dal mazzo infinito)
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # Assi contati come 1
        self.state = None
        self.done = False

    def reset(self):
        """Reset the environment to start a new game."""
        self.done = False
        player_hand = [self.draw_card(), self.draw_card()]
        dealer_hand = [self.draw_card(), self.draw_card()]

        # Lo stato iniziale è la somma delle carte del giocatore, la carta visibile del mazziere, e se c'è un asso
        usable_ace = int(1 in player_hand and sum(player_hand) <= 11)
        self.state = (sum(player_hand), dealer_hand[0], usable_ace)

        return self.state

    def step(self, action):
        """Take a step in the environment based on the action (0 = 'hit', 1 = 'stick')."""
        if self.done:
            return self.state, 0, True, {}

        player_sum, dealer_card, usable_ace = self.state

        if action == 0:  # "hit"
            card = self.draw_card()
            player_sum += card
            # Se abbiamo un asso e la somma è <= 11, trattiamolo come 11
            if usable_ace and player_sum <= 11:
                player_sum += 10
            if player_sum > 21:
                self.done = True
                return (player_sum, dealer_card, usable_ace), -1, True, {}  # Player busts

        # Se il giocatore ha deciso di fermarsi
        if action == 1 or player_sum > 21:
            # Ora il mazziere gioca
            dealer_sum = dealer_card
            while dealer_sum < 17:
                dealer_sum += self.draw_card()
            if dealer_sum > 21:
                self.done = True
                return (player_sum, dealer_card, usable_ace), 1, True, {}  # Dealer busts
            elif dealer_sum > player_sum:
                self.done = True
                return (player_sum, dealer_card, usable_ace), -1, True, {}  # Dealer wins
            elif dealer_sum < player_sum:
                self.done = True
                return (player_sum, dealer_card, usable_ace), 1, True, {}  # Player wins
            else:
                self.done = True
                return (player_sum, dealer_card, usable_ace), 0, True, {}  # Draw

        return self.state, 0, False, {}

    def draw_card(self):
        """Estrai una carta dal mazzo."""
        return random.choice(self.deck)

    def render(self):
        """Render the current state of the environment."""
        player_sum, dealer_card, usable_ace = self.state
        print(f"Player's total: {player_sum} (Ace usable: {usable_ace})")
        print(f"Dealer's face-up card: {dealer_card}")
        print("-------------------------------")

    def close(self):
        """Optionally clean up the environment."""
        pass

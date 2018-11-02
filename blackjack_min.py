'''
This is minimal version of blackjack
for optimization of calculation.
'''

import random
import numpy as np

class Player:

    def __init__(self):
        self.score = 0
        self.ace = 0
        self.hidden = None
        self.hand = []
        self.action = 2

    def best_score(self):
        return self.score + 10 if self.ace and self.score <= 11 else self.score

    def draw(self, card, hidden):
        if hidden:
            self.hidden = card
        else:
            self.hand.append(card)
        self.score += card
        if card == 1:
            self.ace = 1

    def bust(self):
        return self.best_score() == 22

class Agent(Player):

    def decide(self, action):
        self.action = action

class Dealer(Player):

    def decide(self, action):
        self.action = 1 if self.best_score() < 17 else 0

class Deck:

    def __init__(self):
        self.cards = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        self.count = [4] * 9 + [16]
        random.shuffle(self.cards)

    def __len__(self):
        return len(self.cards)

    def draw(self):
        top = self.cards.pop(0)
        self.count[top-1] -= 1
        return top

    def empty(self):
        return len(self) == 0

class Game:

    def __init__(self, agent, dealer, deck=Deck()):
        self.agent = agent
        self.dealer = dealer
        self.players = [self.agent, self. dealer]
        self.deck = Deck() if len(deck) < 4 else deck
        self.ready()

    def ready(self):
        for player in self.players:
            self.draw(player, True)
        for player in self.players:
            self.draw(player)

    def proceed(self, action):
        # 종료조건을 적어주자
        if self.deck.empty():
            return -1   # abnormal exit
        for player in self.players:
            player.d2ecide(action)
            if player.action:
                self.draw(player)


    def draw(self, player, hidden=False):
        top = self.deck.draw()
        player.draw(top, hidden)

    def play(self):
        # return reward
        done = False
        while not done:
            done = self.proceed()
            agent_score = self.agent.best_score()
            dealer_score = self.dealer.best_score()
            if self.agent.bust():
                return -1
            if agent_score > dealer_score:
                return 1
            elif agent_score == dealer_score:
                return 0
            else:
                return -1

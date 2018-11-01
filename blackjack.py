'''
This is BlackJack game enviornment.
This game is only between one agent and one dealer.
'''
import random
import numpy as np
from gym import spaces
import gym

'''
state = [score, ace, bust_prob]
action = [0, 1] 0: stay 1: hit
'''

class Player:

    def __init__(self):
        self.reset()
        self.reward = 0

    def __str__(self):
        return str({'score': self.best_score(), 'hidden': self.hidden, 'hand': self.hand})

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.action = None
        self.score = 0
        self.ace = 0
        self.hidden = None
        self.hand = []
        self.result = None

    def set_score(self, card):
        if card.rank in ['J', 'Q', 'K']:
            self.score += 10
        elif card.rank == 'A':
            self.score += 1
            self.ace = 1
        else:
            self.score += int(card.rank)
        if self.score > 21:    # BUST
            self.score = 22

    def best_score(self):
        '''
        ace까지 포함해서 21이하의 최댓값의 경우를 구해보자
        ace가 11로 쓰이는 건 최대 한번뿐이다.
        '''
        score = self.score
        if self.ace and score < 12:
            score += 10
        return score

    def bust(self):
        return self.score > 21

    def draw(self, deck, hidden=False):
        top = deck.draw()
        if hidden:
            self.hidden = top
        else:
            self.hand.append(top)
        self.set_score(top)
        return top

    def set_state(self, game):
        '''
        1. score ( not including ace )
        2. ace: the number of aces
        (deprecated) 3. counted: cards that the player counted
        3. bust probability:
        '''
        score = self.score  # score는 최대 22
        ace = self.ace  # ace를 가지고 있는지
        counted = [self.hidden] + game.public    # card(suit, rank)들로 해야할까
        unknown = list(set(game.deck.base()) - set(counted))
        bust = 0
        for card in unknown:    # rank 개수로 해야할까
            if card.rank == 'A':
                rank = 1
            elif card.rank in ['J', 'Q', 'K']:
                rank = 10
            else:
                rank = int(card.rank)
            if rank + score > 21:
                bust += 1
        bust_prob = bust / len(unknown) if len(unknown) else 0.5
        self.state = score, ace, bust_prob

class Agent(Player):

    def decide(self, action):
        '''
        action: 0: stay 1: hit
        '''
        self.action = action

class Dealer(Player):

    def decide(self, action):
        self.action = 1 if self.best_score() < 17 else 0

class Card:

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return '%s_%s' % (self.suit, self.rank)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __ne__(self, other):
        return not __eq__(self, other)

    def __hash__(self):
        suit = Card.suits.index(self.suit)
        rank = Card.ranks.index(self.rank)
        return suit*1000 + rank

    suits = ['club', 'diamond', 'heart', 'spade']
    ranks = ['A'] + [str(i) for i in range(2, 11)] + ['J', 'Q', 'K']


class Deck:

    def __init__(self):
        self.cards = self.base()
        random.shuffle(self.cards)

    def count(self):
        return len(self.cards)

    def empty(self):
        return self.count() == 0

    def draw(self):
        return self.cards.pop(0)

    def base(self):
        return [Card(suit, rank) for suit in Card.suits for rank in Card.ranks]

    def public(self):
        return list(set(self.base())-set(self.cards))


class Game:

    def __init__(self, agent, dealer, deck):
        self.agent = agent
        self.dealer = dealer
        self.players = [self.agent, self.dealer]
        for player in self.players:
            player.reset()
        self.deck = deck
        self.player_num = len(self.players)
        self.public = self.deck.public()
        if not self.ready():
            self.deck = Deck()
            self.public = self.deck.public()

    def ready(self):
        if self.deck.count() < 2 * self.player_num:
            return 0
        for player in self.players:
            self.draw(player, hidden=True)
        for player in self.players:
            self.draw(player)
        return 1

    def draw(self, player, hidden=False):
        self.public.append(player.draw(self.deck, hidden))
        self.agent.set_state(self)


class BlackJack(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Box(np.array([0,0,0.0]), np.array([22, 1, 1.0]))
        self.action_space = spaces.Discrete(2)
        self.game = Game(Agent(), Dealer(), Deck())

    def reset(self, deck=Deck()):
        if deck.count() < 4:
            deck = Deck()
        self.game = Game(Agent(), Dealer(), deck)
        # print(self.game.agent)
        return self.game.agent.state

    def step(self, action):
        game = self.game
        reward = 0
        info = {
            'agent': game.agent,
            'dealer': game.dealer
        }

        # whole game ends
        if game.deck.count() < 2:
            return game.agent.state, reward, True, info

        for player in game.players:
            player.decide(action)
            if player.action:
                game.draw(player)

        # agent hit
        if game.agent.action:
            # agent busted : single round end
            if game.agent.bust():
                reward -= 1
                return game.agent.state, reward, True, info

        # agent stay and dealer hit
        if game.agent.action == 0 and game.dealer.action == 1:
            # if agent stays, dealer draws till it stays
            while game.dealer.action:
                if game.deck.empty():
                    return game.agent.state, reward, True, info
                game.draw(game.dealer)
                game.dealer.decide(action)

        if game.agent.action == 0 and game.dealer.action == 0:
            agent_score = game.agent.best_score()
            dealer_score = game.dealer.best_score()
            if game.dealer.bust():
                reward += 1
            else:
                # Win
                if agent_score > dealer_score:
                    reward += 1
                # Lose
                elif agent_score < dealer_score:
                    reward -= 1

            return game.agent.state, reward, True, info

        # round not over
        return game.agent.state, reward, False, info

    def render(self):
        pass


''''
This Blackjack is modified version.
Card should be drawed
'''

import sys
import random
import csv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from blackjack import BlackJack

class Agent:
    
    def __init__(self, state_size, action_size,
                 discount_factor=0.99, learning_rate=0.001,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01,
                 batch_size=64, train_start=1000):
        self.render = False
        # self.load_model = True
        self.load_model = False

        # 상태 & 행동 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # hyper parameter
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start

        # replay memory
        self.memory = deque(maxlen=2000)

        # model & target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # reset target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights('./save_model/blackjack_dqn_trained.h5')

    # input: state, output: Q
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # randomly sample batch from replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [] ,[]

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # update target
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


def learn_agent(env, agent):

    ratios = []
    EPISODES = 50
    SUBEPISODES = 100
    for e in range(EPISODES):
        scores, episodes = [], []
        for se in range(SUBEPISODES):
            done = False
            score = 0
            # reset env
            state = env.reset(deck=env.game.deck)
            state = np.reshape(state, [1, state_size])

            while not done:
                if agent.render:
                    env.render()

                # action by current state
                action = agent.get_action(state)

                # continue time step by chosen action
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                # save sars in replay memory
                agent.append_sample(state, action, reward, next_state, done)

                # learn every time step
                if len(agent.memory) >= agent.train_start:
                    agent.train_model()

                score += reward
                state = next_state

                if done:
                    agent.update_target_model()
                    scores.append(score)
        ratios.append(scores.count(1)/(scores.count(1)+scores.count(-1))*100)

    mean = np.mean(ratios)
    best = max(ratios)
    return mean, best



if __name__ == '__main__' :
    env = BlackJack()    # 여기에 환경을 넣자

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    results = []
    parameters = []
    for i in range(30):
        discount = random.uniform(0.9, 0.9999)
        learn_rate = random.uniform(0.00001, 0.1)
        # DQN agent
        agent = Agent(state_size, action_size, discount_factor=discount, learning_rate=learn_rate)
        results.append(learn_agent(env, agent))
        parameters.append((discount, learn_rate))

    for i in range(10):
        print(results[i], parameters[i])

    # print('Best ratio:', max(ratios))
    # plt.plot([e for e in range(EPISODES)], ratios)
    # plt.show()
    # plt.savefig('result.png')
    # agent.model.save_weights('./save_model/blackjack_dqn_trained.h5')
    # file = open('ratio.csv', 'w+', newline='')
    # wr = csv.writer(file)
    # for e in range(EPISODES):
    #     wr.writerow([e, ratios[e]])

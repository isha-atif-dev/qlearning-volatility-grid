# src/agent.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions=4, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.actions = actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: dictionary mapping state -> list of action values
        self.q_table = defaultdict(lambda: [0.0] * self.actions)

    def choose_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        else:
            return self.q_table[state].index(max(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])

        # Q-learning update formula
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

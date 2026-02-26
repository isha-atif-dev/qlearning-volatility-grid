import random

class QLearningAgent:
    """
    Q-learning agent for a grid world with:
    state = (row, col)
    actions = {0: up, 1: down, 2: left, 3: right}
    """

    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.actions = list(actions)  # e.g. [0,1,2,3]
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # exploration rate
        self.q = {}                   # Q-table: key=(state, action) -> value

    # -------- Q-table helpers --------
    def get_q(self, state, action):
        """Return Q(state, action). Default is 0 if unseen."""
        return self.q.get((state, action), 0.0)

    def best_action(self, state):
        """Return action with highest Q value for this state."""
        best_a = None
        best_q = float("-inf")

        for a in self.actions:
            q_val = self.get_q(state, a)
            if q_val > best_q:
                best_q = q_val
                best_a = a

        return best_a

    # -------- Choose action (epsilon-greedy) --------
    def choose_action(self, state):
        """
        With probability epsilon -> random action (explore)
        Otherwise -> best known action (exploit)
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.best_action(state)

    # -------- Learn from one transition --------
    def update(self, state, action, reward, next_state):
        """
        Q(s,a) = Q(s,a) + alpha * (reward + gamma*max_a' Q(s',a') - Q(s,a))
        """
        old_q = self.get_q(state, action)

        # max future Q
        next_best_q = max(self.get_q(next_state, a) for a in self.actions)

        # target
        target = reward + self.gamma * next_best_q

        # update
        new_q = old_q + self.alpha * (target - old_q)
        self.q[(state, action)] = new_q

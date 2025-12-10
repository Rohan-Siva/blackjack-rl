import numpy as np
import pickle
import collections

class MCAgent:
    def __init__(self, action_num=2, epsilon=0.1, gamma=0.99):
        self.action_num = action_num
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = collections.defaultdict(float)
        self.return_sum = collections.defaultdict(float)
        self.return_count = collections.defaultdict(float)
        
    def step(self, state):
        state_key = self._get_state_key(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_num)
        
        q_values = [self.Q[(state_key, a)] for a in range(self.action_num)]
        if all(q == 0 for q in q_values):
            return np.random.randint(self.action_num)
            
        # we break ties randomly
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def eval_step(self, state):
        state_key = self._get_state_key(state)
        q_values = [self.Q[(state_key, a)] for a in range(self.action_num)]
        
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

    def update(self, episode_transitions):
        G = 0
        visited = set()
        
        for state, action, reward in reversed(episode_transitions):
            state_key = self._get_state_key(state)
            G = self.gamma * G + reward
            
            sa_pair = (state_key, action)
            
            if sa_pair not in visited:
                visited.add(sa_pair)
                self.return_sum[sa_pair] += G
                self.return_count[sa_pair] += 1
                self.Q[sa_pair] = self.return_sum[sa_pair] / self.return_count[sa_pair]

    def _get_state_key(self, state):
        if isinstance(state, dict):
            state = state['obs']
        return tuple(state)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.Q), f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.Q = update_defaultdict(pickle.load(f))

def update_defaultdict(d):
    new_d = collections.defaultdict(float)
    new_d.update(d)
    return new_d

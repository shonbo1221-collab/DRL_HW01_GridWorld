import numpy as np
import random

def get_next_state_reward(n, s, a, end_idx, obstacles):
    if s == end_idx:
        return s, 0.0
    if s in obstacles:
        return s, 0.0
        
    r, c = divmod(s, n)
    if a == 0:   # Up
        r = max(0, r - 1)
    elif a == 1: # Down
        r = min(n - 1, r + 1)
    elif a == 2: # Left
        c = max(0, c - 1)
    elif a == 3: # Right
        c = min(n - 1, c + 1)
        
    next_s = r * n + c
    if next_s in obstacles:
        next_s = s
        
    reward = 10.0 if next_s == end_idx else -1.0
    return next_s, reward


def evaluate_policy(n, end_idx, obstacles, gamma=0.9, theta=1e-4):
    total_cells = n * n
    V = np.zeros(total_cells)
    
    actions = [0, 1, 2, 3]
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    policy = {}
    for s in range(total_cells):
        if s != end_idx and s not in obstacles:
            policy[s] = random.choice(actions)
            
    while True:
        delta = 0
        for s in range(total_cells):
            if s == end_idx or s in obstacles:
                continue
            v = V[s]
            a = policy[s]
            next_s, reward = get_next_state_reward(n, s, a, end_idx, obstacles)
            new_v = reward + gamma * V[next_s]
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
            
        if delta < theta:
            break
            
    str_policy = {k: arrow_map[v] for k, v in policy.items()}
    values = [round(val, 3) for val in V.tolist()]
    return str_policy, values


def value_iteration(n, end_idx, obstacles, gamma=0.9, theta=1e-4):
    total_cells = n * n
    V = np.zeros(total_cells)
    
    actions = [0, 1, 2, 3]
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    # 1. Value Iteration (Loop until V converges)
    while True:
        delta = 0
        new_V = np.copy(V)
        for s in range(total_cells):
            if s == end_idx or s in obstacles:
                continue
                
            v = V[s]
            action_values = []
            for a in actions:
                next_s, reward = get_next_state_reward(n, s, a, end_idx, obstacles)
                action_values.append(reward + gamma * V[next_s])
                
            best_value = max(action_values)
            new_V[s] = best_value
            delta = max(delta, abs(v - best_value))
            
        V = new_V
        if delta < theta:
            break

    # 2. Extract Optimal Policy
    policy = {}
    for s in range(total_cells):
        if s == end_idx or s in obstacles:
            continue
            
        action_values = []
        for a in actions:
            next_s, reward = get_next_state_reward(n, s, a, end_idx, obstacles)
            action_values.append(reward + gamma * V[next_s])
            
        best_a = actions[np.argmax(action_values)]
        policy[s] = arrow_map[best_a]
        
    values = [round(val, 3) for val in V.tolist()]
    return policy, values

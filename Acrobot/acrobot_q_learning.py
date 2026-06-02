import gymnasium as gym
import numpy as np
import pickle
import os

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1
EPISODES = 2000
RENDER_EVERY = 500

# Create the Acrobot environment
env = gym.make("Acrobot-v1")

# Discretization setup
# Observation space: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
# Ranges (approx): [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-12.5, 12.5], [-28.2, 28.2]
num_bins = [10, 10, 10, 10, 10, 10]
obs_low = np.array([-1, -1, -1, -1, -12.5, -28.2])
obs_high = np.array([1, 1, 1, 1, 12.5, 28.2])

def discretize_state(state):
    state_adj = (state - obs_low) / (obs_high - obs_low)
    state_adj = np.clip(state_adj, 0, 1)
    discretized = (state_adj * (np.array(num_bins) - 1)).astype(int)
    return tuple(discretized)

# Initialize Q-table
q_table_path = "Acrobot/acrobot_q_table.pkl"
if os.path.exists(q_table_path):
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    print("Loaded existing Q-table.")
else:
    q_table = np.zeros(num_bins + [env.action_space.n])
    print("Initialized new Q-table.")

def train():
    global EPSILON
    for episode in range(EPISODES):
        state, info = env.reset()
        state = discretize_state(state)
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if np.random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_state_raw)
            
            # Q-Learning update
            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
            q_table[state + (action,)] = new_value
            
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{EPISODES} | Reward: {total_reward} | Epsilon: {EPSILON:.2f}")
        
        # Decay epsilon
        if EPSILON > 0.01:
            EPSILON *= 0.995

    # Save the Q-table
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
    print("Training finished. Q-table saved.")

def test():
    print("Testing the trained agent...")
    test_env = gym.make("Acrobot-v1", render_mode="human")
    state, info = test_env.reset()
    state = discretize_state(state)
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        next_state_raw, reward, terminated, truncated, info = test_env.step(action)
        state = discretize_state(next_state_raw)
        total_reward += reward
    
    print(f"Test Episode Reward: {total_reward}")
    test_env.close()

if __name__ == "__main__":
    train()
    test()

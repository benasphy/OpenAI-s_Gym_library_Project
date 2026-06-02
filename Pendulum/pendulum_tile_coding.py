import gymnasium as gym
import numpy as np
import pickle
import os

# Tile Coding Parameters
NUM_TILINGS = 8
BINS = 10
# Observation: [cos(theta), sin(theta), theta_dot]
# Ranges: [-1, 1], [-1, 1], [-8, 8]
OBS_LOW = np.array([-1.0, -1.0, -8.0])
OBS_HIGH = np.array([1.0, 1.0, 8.0])

# Hyperparameters
ALPHA = 0.1 / NUM_TILINGS  # Learning rate split across tilings
EPSILON = 0.1
GAMMA = 0.99
EPISODES = 2000

class TileCoder:
    def __init__(self, num_tilings, bins, obs_low, obs_high, num_actions):
        self.num_tilings = num_tilings
        self.bins = bins
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.num_actions = num_actions
        
        # Calculate offsets for each tiling: (num_tilings, num_dims)
        self.offsets = (np.arange(num_tilings)[:, np.newaxis] / num_tilings) * (obs_high - obs_low) / bins
        
        # Weights for the linear model: (num_tilings, bins+1, bins+1, bins+1, num_actions)
        self.weights = np.zeros((num_tilings, bins + 1, bins + 1, bins + 1, num_actions))

    def get_features(self, state):
        features = []
        for i in range(self.num_tilings):
            # Apply offset and normalize
            normalized_state = (state + self.offsets[i] - self.obs_low) / (self.obs_high - self.obs_low)
            # Map to bins
            tiles = (normalized_state * self.bins).astype(int)
            # Clip to stay within weight array bounds
            tiles = np.clip(tiles, 0, self.bins)
            features.append(tuple([i] + list(tiles)))
        return features

    def get_q_values(self, state):
        features = self.get_features(state)
        q_values = np.zeros(self.num_actions)
        for feat in features:
            q_values += self.weights[feat]
        return q_values

    def update(self, state, action, target):
        features = self.get_features(state)
        q_values = self.get_q_values(state)
        error = target - q_values[action]
        for feat in features:
            self.weights[feat + (action,)] += ALPHA * error

# Discrete actions for Pendulum: -2.0, 0, 2.0
ACTIONS = np.array([-2.0, 0.0, 2.0])
NUM_ACTIONS = len(ACTIONS)

def train():
    env = gym.make("Pendulum-v1")
    agent = TileCoder(NUM_TILINGS, BINS, OBS_LOW, OBS_HIGH, NUM_ACTIONS)
    
    q_table_path = "Pendulum/pendulum_weights.pkl"
    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as f:
            agent.weights = pickle.load(f)
        print("Loaded existing weights.")

    print(f"Training for {EPISODES} episodes...")
    for ep in range(EPISODES):
        state, info = env.reset()
        total_reward = 0
        
        # Epsilon-greedy
        if np.random.random() < EPSILON:
            action_idx = np.random.randint(NUM_ACTIONS)
        else:
            action_idx = np.argmax(agent.get_q_values(state))
            
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = [ACTIONS[action_idx]]
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # SARSA Update
            if np.random.random() < EPSILON:
                next_action_idx = np.random.randint(NUM_ACTIONS)
            else:
                next_action_idx = np.argmax(agent.get_q_values(next_state))
                
            target = reward + GAMMA * agent.get_q_values(next_state)[next_action_idx]
            agent.update(state, action_idx, target)
            
            state = next_state
            action_idx = next_action_idx
            total_reward += reward
            
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{EPISODES} | Reward: {total_reward:.2f}")

    with open(q_table_path, "wb") as f:
        pickle.dump(agent.weights, f)
    print("Training finished. Weights saved.")
    env.close()
    return agent

def test(agent=None):
    if agent is None:
        agent = TileCoder(NUM_TILINGS, BINS, OBS_LOW, OBS_HIGH, NUM_ACTIONS)
        with open("Pendulum/pendulum_weights.pkl", "rb") as f:
            agent.weights = pickle.load(f)
            
    print("Testing the agent...")
    env = gym.make("Pendulum-v1", render_mode="human")
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action_idx = np.argmax(agent.get_q_values(state))
        action = [ACTIONS[action_idx]]
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Test Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    import sys
    
    q_table_path = "Pendulum/pendulum_weights.pkl"
    
    if "--train" in sys.argv or not os.path.exists(q_table_path):
        trained_agent = train()
        test(trained_agent)
    else:
        test()

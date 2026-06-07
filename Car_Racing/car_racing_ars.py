import gymnasium as gym
import numpy as np
import pickle
import os
import sys

# Hyperparameters for ARS (specifically tuned for Car Racing)
HP = {
    "n_directions": 8,      # Reduced directions for faster steps with images
    "n_best": 4,            # Number of best directions to keep
    "step_size": 0.05,      # Learning rate
    "noise": 0.03,          # Standard deviation of the noise
    "episodes": 1000,       # Total training episodes
    "episode_length": 1000, # Max steps per episode
}

def preprocess(obs):
    # Convert to grayscale and downsample
    # Original is 96x96x3. We'll take 48x48 grayscale.
    gray = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
    downsampled = gray[::2, ::2]
    return downsampled.flatten() / 255.0

class Normalizer:
    def __init__(self, n_inputs):
        self.n = np.zeros(n_inputs)
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - last_mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Policy:
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta @ input
        elif direction == "positive":
            return (self.theta + HP["noise"] * delta) @ input
        else:
            return (self.theta - HP["noise"] * delta) @ input

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += HP["step_size"] / (HP["n_best"] * sigma_r) * step

def explore(env, normalizer, policy, direction=None, delta=None):
    state, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < HP["episode_length"]:
        state_pre = preprocess(state)
        normalizer.observe(state_pre)
        state_norm = normalizer.normalize(state_pre)
        action = policy.evaluate(state_norm, delta, direction)
        
        # Action space for CarRacing: [steer, gas, break]
        # steer: [-1, 1], gas: [0, 1], break: [0, 1]
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.0, 1.0)
        action[2] = np.clip(action[2], 0.0, 1.0)
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return total_reward

def train():
    env = gym.make("CarRacing-v3")
    # Preprocessed input size: 48 * 48 = 2304
    input_size = 2304
    output_size = 3
    
    policy = Policy(input_size, output_size)
    normalizer = Normalizer(input_size)
    
    weights_path = "Car_Racing/car_racing_ars.pkl"
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            policy.theta, normalizer.mean, normalizer.var = pickle.load(f)
        print("Loaded existing weights.")

    print(f"Training ARS for Car Racing ({HP['episodes']} episodes)...")
    for step in range(HP["episodes"]):
        deltas = [np.random.randn(*policy.theta.shape) for _ in range(HP["n_directions"])]
        pos_rewards = [0] * HP["n_directions"]
        neg_rewards = [0] * HP["n_directions"]

        for i in range(HP["n_directions"]):
            pos_rewards[i] = explore(env, normalizer, policy, direction="positive", delta=deltas[i])
            neg_rewards[i] = explore(env, normalizer, policy, direction="negative", delta=deltas[i])

        scores = {i: max(pos_rewards[i], neg_rewards[i]) for i in range(HP["n_directions"])}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:HP["n_best"]]
        rollouts = [(pos_rewards[k], neg_rewards[k], deltas[k]) for k in order]

        sigma_r = np.std(np.array([r for r in pos_rewards + neg_rewards]))
        if sigma_r > 0:
            policy.update(rollouts, sigma_r)

        if (step + 1) % 5 == 0:
            reward_test = explore(env, normalizer, policy)
            print(f"Step {step + 1}/{HP['episodes']} | Reward: {reward_test:.2f}")
            with open(weights_path, "wb") as f:
                pickle.dump((policy.theta, normalizer.mean, normalizer.var), f)

    print("Training finished. Weights saved.")
    env.close()

def test():
    env = gym.make("CarRacing-v3", render_mode="human")
    input_size = 2304
    output_size = 3
    policy = Policy(input_size, output_size)
    normalizer = Normalizer(input_size)
    
    weights_path = "Car_Racing/car_racing_ars.pkl"
    if not os.path.exists(weights_path):
        print("No weights found. Please train first.")
        return

    with open(weights_path, "rb") as f:
        policy.theta, normalizer.mean, normalizer.var = pickle.load(f)
    
    print("Testing the trained agent...")
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_pre = preprocess(state)
        state_norm = normalizer.normalize(state_pre)
        action = policy.evaluate(state_norm)
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.0, 1.0)
        action[2] = np.clip(action[2], 0.0, 1.0)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"Test Final Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    if "--train" in sys.argv or not os.path.exists("Car_Racing/car_racing_ars.pkl"):
        train()
        test()
    else:
        test()

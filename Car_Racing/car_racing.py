import gymnasium as gym

# Create the Car Racing environment
# This is a Box2D environment where a car needs to navigate a track.
# State space: 96x96 pixels (RGB)
# Action space: Continuous (steer, gas, break)
env = gym.make("CarRacing-v3", render_mode="human")

observation, info = env.reset()

print(f"Starting observation shape: {observation.shape}")

episode_over = False
total_reward = 0

print("Running Car Racing with random actions...")

try:
    while not episode_over:
        # Choose a random action: [steer, gas, break]
        # steer: [-1.0, 1.0], gas: [0.0, 1.0], break: [0.0, 1.0]
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
except KeyboardInterrupt:
    print("\nExecution interrupted by user.")

print(f"Episode finished! Total reward: {total_reward}")
env.close()

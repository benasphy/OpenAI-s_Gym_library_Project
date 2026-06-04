import gymnasium as gym

# Create the Bipedal Walker Hardcore environment
# This is a Box2D environment where a robot needs to walk across difficult terrain
# including stairs, stumps, and pits.
# State space: 24 dimensions (lidar, angles, velocities)
# Action space: 4 continuous values (torque for 4 joints)
env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

print("Running Bipedal Walker Hardcore with random actions...")

try:
    while not episode_over:
        # Choose a random action: 4 torques in the range [-1.0, 1.0]
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
except KeyboardInterrupt:
    print("\nExecution interrupted by user.")

print(f"Episode finished! Total reward: {total_reward}")
env.close()

import gymnasium as gym

# Create the Pendulum environment
# The goal is to keep a frictionless pendulum standing upright.
# This environment has a continuous action space (torque).
env = gym.make("Pendulum-v1", render_mode="human")

# Reset the environment
observation, info = env.reset()

print(f"Starting observation: {observation}")
# Observation: [cos(theta), sin(theta), theta_dot]

episode_over = False
total_reward = 0

print("Running Pendulum with random actions...")

try:
    while not episode_over:
        # Choose a random action: torque in the range [-2.0, 2.0]
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
except KeyboardInterrupt:
    print("\nExecution interrupted by user.")

print(f"Episode finished! Total reward: {total_reward}")
env.close()

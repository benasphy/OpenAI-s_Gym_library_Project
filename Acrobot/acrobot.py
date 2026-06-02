import gymnasium as gym

# Create the Acrobot environment
# Acrobot consists of two links and two joints, where only the second joint is actuated.
# The goal is to swing the end of the lower link up to a given height.
env = gym.make("Acrobot-v1", render_mode="human")

# Reset the environment to start a new episode
observation, info = env.reset()

print(f"Starting observation: {observation}")
# Observation space usually includes:
# [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]

episode_over = False
total_reward = 0

print("Running Acrobot with random actions...")

try:
    while not episode_over:
        # Choose a random action: 0 (apply -1 torque), 1 (apply 0 torque), 2 (apply +1 torque)
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
        
        # The environment renders automatically in "human" mode when step() is called
except KeyboardInterrupt:
    print("\nExecution interrupted by user.")

print(f"Episode finished! Total reward: {total_reward}")
env.close()

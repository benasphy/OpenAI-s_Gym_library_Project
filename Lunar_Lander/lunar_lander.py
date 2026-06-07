import gymnasium as gym

# Create the Lunar Lander environment
# Goal: Land the lander on the landing pad (at coordinates 0,0)
# State space: 8 variables (x, y, vx, vy, angle, v_angle, left_leg, right_leg)
# Action space: Discrete(4) (0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine)
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

print("Running Lunar Lander with random actions...")

try:
    while not episode_over:
        # Choose a random action (0-3)
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
except KeyboardInterrupt:
    print("\nExecution interrupted by user.")

print(f"Episode finished! Total reward: {total_reward}")
env.close()

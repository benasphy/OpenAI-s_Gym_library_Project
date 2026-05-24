import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def _greedy_action(q_row, rng, random_tiebreak=False):
    if not random_tiebreak:
        return int(np.argmax(q_row))
    max_q = np.max(q_row)
    max_actions = np.flatnonzero(np.isclose(q_row, max_q))
    return int(rng.choice(max_actions))


def train(episodes, is_8x8=False, is_slippery=True, q_table=None):
    """Training mode: Uses epsilon-greedy exploration and updates Q-table"""
    map_name = "8x8" if is_8x8 else "4x4"
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    
    # Initialize Q-table if not provided
    q = q_table if q_table is not None else np.zeros((env.observation_space.n, env.action_space.n))
    
    # Adaptive learning rate - starts high, decays over time for stability
    learning_rate_a = 0.3 if is_8x8 else 0.8
    discount_factor_g = 0.95
    
    # Exploration schedule:
    # - Keep existing behavior for deterministic mode.
    # - Use slower decay and higher min epsilon for slippery mode.
    epsilon = 1.0
    if is_slippery:
        epsilon_decay_rate = 0.00003 if is_8x8 else 0.00015
        min_epsilon = 0.05 if is_8x8 else 0.02
    else:
        epsilon_decay_rate = 0.0001 if is_8x8 else 0.0005
        min_epsilon = 0.01
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)
    
    goal_state = env.observation_space.n - 1
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        # Adaptive learning rate: decay over time
        current_lr = learning_rate_a * (1 - i / episodes * 0.5)
        
        while not terminated and not truncated and steps < 500:
            # TRAINING MODE: epsilon-greedy exploration
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = _greedy_action(q[state,:], rng, random_tiebreak=is_slippery)
                
            new_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            
            # Reward shaping - smoother penalties
            shaped_reward = reward + (0.05 if new_state == goal_state else -0.005)
            
            # Q-table UPDATE with adaptive learning rate (more stable)
            q[state, action] = q[state, action] + current_lr * (
                shaped_reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
            )
            
            state = new_state
            episode_reward += reward
        
        # Decay exploration - smoother decay
        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
        
        if episode_reward == 1:
            rewards_per_episode[i] = 1
        steps_per_episode[i] = steps
    
    env.close()
    
    return q, rewards_per_episode, steps_per_episode


def test(episodes, q_table, is_8x8=False, is_slippery=True, render=False, eval_epsilon=None):
    """Inference mode: Uses learned Q-table with optional tiny epsilon for slippery mode."""
    map_name = "8x8" if is_8x8 else "4x4"
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human' if render else None)

    # Keep existing deterministic behavior by default.
    # For slippery mode, a tiny eval epsilon can reduce local looping.
    if eval_epsilon is None:
        eval_epsilon = 0.01 if is_slippery else 0.0
    rng = np.random.default_rng()
    
    q = q_table.copy()
    
    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not terminated and not truncated and steps < 500:
            if rng.random() < eval_epsilon:
                action = env.action_space.sample()
            else:
                action = _greedy_action(q[state,:], rng, random_tiebreak=is_slippery)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            
            state = new_state
            episode_reward += reward
        
        if episode_reward == 1:
            rewards_per_episode[i] = 1
        steps_per_episode[i] = steps
    
    env.close()
    return rewards_per_episode, steps_per_episode


def run(episodes, is_8x8=False, is_slippery=False, render=False, eval_epsilon=None):
    """Run complete training and testing pipeline"""
    q, rewards_train, steps_train = train(episodes, is_8x8=is_8x8, is_slippery=is_slippery)
    print(f"Training complete. Testing learned policy on {episodes} episodes...")
    rewards_test, steps_test = test(
        episodes,
        q,
        is_8x8=is_8x8,
        is_slippery=is_slippery,
        render=render,
        eval_epsilon=eval_epsilon,
    )
    mode = "slippery" if is_slippery else "deterministic"
    visualize_results(rewards_train, steps_train, q, f"{'8x8' if is_8x8 else '4x4'} FrozenLake ({mode})")


def visualize_results(rewards_per_episode, steps_per_episode, q, title="Training Results"):
    """Visualize learning metrics"""
    episodes = len(rewards_per_episode)
    window_size = 50
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-window_size):(t+1)])
    
    # Better visualization
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Success rate
    plt.subplot(1, 3, 1)
    plt.plot(sum_rewards / window_size * 100, linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Success Rate (%)', fontsize=11)
    plt.title('Success Rate Over Time', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.fill_between(range(episodes), 0, sum_rewards / window_size * 100, alpha=0.2)
    
    # Plot 2: Cumulative wins
    plt.subplot(1, 3, 2)
    cumulative = np.cumsum(rewards_per_episode)
    plt.plot(cumulative, linewidth=2, color='green')
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Cumulative Wins', fontsize=11)
    plt.title('Cumulative Successful Episodes', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Steps per episode
    plt.subplot(1, 3, 3)
    plt.plot(steps_per_episode, alpha=0.5, label='Steps per episode', color='red')
    window_steps = np.zeros(episodes)
    for t in range(episodes):
        window_steps[t] = np.mean(steps_per_episode[max(0, t-window_size):(t+1)])
    plt.plot(window_steps, linewidth=2, label='Smoothed average', color='darkred')
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Steps', fontsize=11)
    plt.title('Episode Length Over Time', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('frozen_lake_result.png', dpi=150)
    
    # Performance metrics
    final_success_rate = np.sum(rewards_per_episode[-100:]) / 100 * 100
    avg_steps_final = np.mean(steps_per_episode[-100:])
    print(f"\n{'='*50}")
    print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
    print(f"Average steps (last 100 episodes): {avg_steps_final:.1f}")
    print(f"Total wins: {int(np.sum(rewards_per_episode))} / {episodes}")
    print(f"{'='*50}\n")
    
    f = open("frozen_lake8x8.pkl", "wb")
    pickle.dump(q, f)
    f.close()


if __name__ == "__main__":
    is_8x8 = True
    is_slippery = True

    # Keep existing strategy for deterministic mode.
    # Use more episodes when slippery mode is enabled.
    episodes = 150000 if is_slippery and is_8x8 else 50000

    print(f"Training on {'8x8' if is_8x8 else '4x4'} FrozenLake | slippery={is_slippery}...")
    q, rewards_train, steps_train = train(episodes, is_8x8=is_8x8, is_slippery=is_slippery)
    print("Training complete. Visualizing...")
    mode = "slippery" if is_slippery else "deterministic"
    visualize_results(rewards_train, steps_train, q, f"{'8x8' if is_8x8 else '4x4'} FrozenLake ({mode})")
    print("\nRunning 10 episodes with rendering using learned Q-table...")
    test(
        10,
        q,
        is_8x8=is_8x8,
        is_slippery=is_slippery,
        render=True,
        eval_epsilon=(0.01 if is_slippery else 0.0),
    )
        
        
    
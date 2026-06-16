import pickle
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def train():
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    # Create environment and agent
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    print(f"Training for {n_episodes} episodes...")
    for episode in range(n_episodes):
        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    # Save the agent's Q-values
    out_dir = Path(__file__).parent
    q_table_path = out_dir / "blackjack_q_table.pkl"
    # defaultdict needs to be converted to dict for pickling if the lambda is used
    with open(q_table_path, "wb") as f:
        pickle.dump(dict(agent.q_values), f)
    print(f"Saved Q-table to {q_table_path}")

    # Visualize results
    print("Visualizing results...")
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plot_path = out_dir / "blackjack_result.png"
    plt.savefig(plot_path)
    print(f"Saved results plot to {plot_path}")
    # plt.show() # Disabled for headless environments

    return agent


def test(agent=None):
    print("\nTesting the trained agent (10 episodes)...")
    if agent is None:
        # Load the agent if not provided
        out_dir = Path(__file__).parent
        q_table_path = out_dir / "blackjack_q_table.pkl"
        if not q_table_path.exists():
            print("No trained agent found. Please train first.")
            return
        
        with open(q_table_path, "rb") as f:
            q_values = pickle.load(f)
        
        # Create a dummy agent to hold the loaded Q-values
        env = gym.make("Blackjack-v1", render_mode="human")
        agent = BlackjackAgent(env, 0, 0, 0, 0)
        agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), q_values)
        agent.epsilon = 0.0 # No exploration during testing
    else:
        env = gym.make("Blackjack-v1", render_mode="human")
        agent.env = env
        agent.epsilon = 0.0

    for i in range(10):
        obs, info = env.reset()
        done = False
        print(f"Episode {i+1}: ", end="")
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if reward > 0:
            print("WIN")
        elif reward < 0:
            print("LOSE")
        else:
            print("DRAW")
    
    env.close()


if __name__ == "__main__":
    trained_agent = train()
    # test(trained_agent) # Uncomment to run demo with rendering

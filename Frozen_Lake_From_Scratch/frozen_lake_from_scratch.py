import time
import pickle
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class DiscreteSpace:
    """Simple discrete space replacement for Gym's Discrete."""

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.integers(0, self.n))


class FrozenLakeEnv:
    """Frozen Lake environment implemented from scratch (no Gym dependency)."""

    MAPS = {
        "4x4": [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        ],
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ],
    }

    # Match Gym FrozenLake action indexing: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    ACTION_DELTAS = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
    }

    def __init__(
        self,
        map_name: str = "4x4",
        is_slippery: bool = False,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        if map_name not in self.MAPS:
            raise ValueError(f"Unsupported map_name: {map_name}")

        self.desc = np.array([list(row) for row in self.MAPS[map_name]], dtype="U1")
        self.nrow, self.ncol = self.desc.shape
        self.is_slippery = is_slippery
        self.max_steps = max_steps

        self._rng = np.random.default_rng(seed)
        self.observation_space = DiscreteSpace(self.nrow * self.ncol, self._rng)
        self.action_space = DiscreteSpace(4, self._rng)

        self.start_pos = self._find_tile("S")
        self.goal_pos = self._find_tile("G")

        self.agent_pos = self.start_pos
        self.steps_taken = 0

        # GUI renderer state (created lazily on first gui render call).
        self._fig = None
        self._ax = None
        self._img = None
        self._agent_text = None

    def _find_tile(self, tile: str) -> Tuple[int, int]:
        positions = np.argwhere(self.desc == tile)
        if len(positions) != 1:
            raise ValueError(f"Map must contain exactly one '{tile}' tile")
        return tuple(int(v) for v in positions[0])

    def _to_state(self, row: int, col: int) -> int:
        return row * self.ncol + col

    def _from_state(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.ncol)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space = DiscreteSpace(4, self._rng)
            self.observation_space = DiscreteSpace(self.nrow * self.ncol, self._rng)

        self.agent_pos = self.start_pos
        self.steps_taken = 0
        state = self._to_state(*self.agent_pos)
        return state, {}

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dr, dc = self.ACTION_DELTAS[action]
        nr = min(max(pos[0] + dr, 0), self.nrow - 1)
        nc = min(max(pos[1] + dc, 0), self.ncol - 1)
        return nr, nc

    def _sample_slippery_action(self, action: int) -> int:
        # In slippery mode, move intended direction or one perpendicular direction.
        if action in (0, 2):
            candidates = [action, 3, 1]  # left/right with up/down slips
        else:
            candidates = [action, 0, 2]  # up/down with left/right slips
        return int(self._rng.choice(candidates))

    def step(self, action: int):
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be one of 0,1,2,3")

        self.steps_taken += 1

        chosen_action = self._sample_slippery_action(action) if self.is_slippery else action
        self.agent_pos = self._apply_action(self.agent_pos, chosen_action)

        r, c = self.agent_pos
        tile = self.desc[r, c]

        terminated = tile in ("H", "G")
        truncated = self.steps_taken >= self.max_steps and not terminated
        reward = 1.0 if tile == "G" else 0.0

        next_state = self._to_state(r, c)
        info = {"tile": tile, "chosen_action": chosen_action}
        return next_state, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        grid = self.desc.copy()
        ar, ac = self.agent_pos
        grid[ar, ac] = "A"

        lines = [" ".join(row.tolist()) for row in grid]
        output = "\n".join(lines)

        if mode == "ansi":
            return output

        if mode == "human_text":
            print("\033[H\033[J", end="")
            print(output)
            print(f"steps={self.steps_taken}, pos={self.agent_pos}")
            return None

        if mode == "human":
            self._render_gui()
            return None

        raise ValueError("mode must be 'human', 'human_text', or 'ansi'")

    def _render_gui(self):
        # Tile encoding: S=0, F=1, H=2, G=3.
        tile_to_int = {"S": 0, "F": 1, "H": 2, "G": 3}
        encoded = np.vectorize(tile_to_int.get)(self.desc)

        if self._fig is None or self._ax is None or self._img is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            cmap = ListedColormap([
                "#8fd3ff",  # S
                "#dff6dd",  # F
                "#2b2b2b",  # H
                "#ffe066",  # G
            ])
            self._img = self._ax.imshow(encoded, cmap=cmap, vmin=0, vmax=3)

            self._ax.set_xticks(np.arange(-0.5, self.ncol, 1), minor=True)
            self._ax.set_yticks(np.arange(-0.5, self.nrow, 1), minor=True)
            self._ax.grid(which="minor", color="white", linewidth=1.5)
            self._ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            for r in range(self.nrow):
                for c in range(self.ncol):
                    self._ax.text(c, r, self.desc[r, c], ha="center", va="center", color="black", fontsize=9)

            self._agent_text = self._ax.text(
                self.agent_pos[1],
                self.agent_pos[0],
                "A",
                ha="center",
                va="center",
                color="crimson",
                fontsize=13,
                fontweight="bold",
            )
        else:
            self._img.set_data(encoded)
            self._agent_text.set_position((self.agent_pos[1], self.agent_pos[0]))

        self._ax.set_title(f"FrozenLake | steps={self.steps_taken} | pos={self.agent_pos}")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._img = None
            self._agent_text = None


def train(episodes: int, is_8x8: bool = False, is_slippery: bool = False, q_table=None):
    map_name = "8x8" if is_8x8 else "4x4"
    env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery, max_steps=500)

    q = q_table if q_table is not None else np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate_a = 0.3 if is_8x8 else 0.8
    discount_factor_g = 0.95

    epsilon = 1.0
    epsilon_decay_rate = 0.0001 if is_8x8 else 0.0005
    min_epsilon = 0.01

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    goal_state = env.observation_space.n - 1

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        current_lr = learning_rate_a * (1 - i / episodes * 0.5)

        while not terminated and not truncated and steps < env.max_steps:
            if env._rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[state, :]))

            new_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1

            shaped_reward = reward + (0.05 if new_state == goal_state else -0.005)

            q[state, action] = q[state, action] + current_lr * (
                shaped_reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state
            episode_reward += reward

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        rewards_per_episode[i] = 1 if episode_reward == 1 else 0
        steps_per_episode[i] = steps

    return q, rewards_per_episode, steps_per_episode


def test(
    episodes: int,
    q_table,
    is_8x8: bool = False,
    is_slippery: bool = False,
    render: bool = False,
    render_mode: str = "human",
    render_delay: float = 0.12,
):
    map_name = "8x8" if is_8x8 else "4x4"
    env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery, max_steps=500)

    q = q_table.copy()

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        if render:
            print(f"Episode {i + 1}/{episodes}")
            env.render(render_mode)
            time.sleep(render_delay)

        while not terminated and not truncated and steps < env.max_steps:
            action = int(np.argmax(q[state, :]))
            new_state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            state = new_state
            episode_reward += reward

            if render:
                env.render(render_mode)
                print(f"action={action}, chosen_action={info['chosen_action']}, reward={reward}")
                time.sleep(render_delay)

        rewards_per_episode[i] = 1 if episode_reward == 1 else 0
        steps_per_episode[i] = steps

        if render:
            print("Result:", "WIN" if episode_reward == 1 else "LOSS")
            time.sleep(0.6)

    env.close()
    return rewards_per_episode, steps_per_episode


def visualize_results(rewards_per_episode, steps_per_episode, q, title="Training Results"):
    episodes = len(rewards_per_episode)
    window_size = 50
    sum_rewards = np.zeros(episodes)

    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - window_size):(t + 1)])

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(sum_rewards / window_size * 100, linewidth=2, color="blue")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Success Rate (%)", fontsize=11)
    plt.title("Success Rate Over Time", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.fill_between(range(episodes), 0, sum_rewards / window_size * 100, alpha=0.2)

    plt.subplot(1, 3, 2)
    cumulative = np.cumsum(rewards_per_episode)
    plt.plot(cumulative, linewidth=2, color="green")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Cumulative Wins", fontsize=11)
    plt.title("Cumulative Successful Episodes", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(steps_per_episode, alpha=0.5, label="Steps per episode", color="red")
    window_steps = np.zeros(episodes)
    for t in range(episodes):
        window_steps[t] = np.mean(steps_per_episode[max(0, t - window_size):(t + 1)])
    plt.plot(window_steps, linewidth=2, label="Smoothed average", color="darkred")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Steps", fontsize=11)
    plt.title("Episode Length Over Time", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("frozen_lake_from_scratch_result.png", dpi=150)

    final_success_rate = np.sum(rewards_per_episode[-100:]) / 100 * 100
    avg_steps_final = np.mean(steps_per_episode[-100:])

    print("\n" + "=" * 55)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
    print(f"Average steps (last 100 episodes): {avg_steps_final:.1f}")
    print(f"Total wins: {int(np.sum(rewards_per_episode))} / {episodes}")
    print("=" * 55 + "\n")

    with open("frozen_lake_from_scratch_qtable.pkl", "wb") as f:
        pickle.dump(q, f)


def run(episodes: int, is_8x8: bool = False, is_slippery: bool = False, render_test: bool = False):
    q, rewards_train, steps_train = train(
        episodes=episodes,
        is_8x8=is_8x8,
        is_slippery=is_slippery,
    )
    print(f"Training complete. Testing learned policy on {episodes} episodes...")
    rewards_test, steps_test = test(
        episodes=episodes,
        q_table=q,
        is_8x8=is_8x8,
        is_slippery=is_slippery,
        render=False,
    )

    mode_label = "8x8" if is_8x8 else "4x4"
    slip_label = "slippery" if is_slippery else "deterministic"
    visualize_results(rewards_train, steps_train, q, f"{mode_label} FrozenLake ({slip_label})")

    if render_test:
        print("Running 5 rendered episodes using the learned policy...")
        test(5, q, is_8x8=is_8x8, is_slippery=is_slippery, render=True)

    return q, rewards_train, steps_train, rewards_test, steps_test


if __name__ == "__main__":
    # Same spirit as your Gym script, now fully standalone.
    print("Training custom 8x8 FrozenLake (no Gym)...")
    q, rewards_train, steps_train = train(50000, is_8x8=True, is_slippery=False)
    print("Training complete. Visualizing...")
    visualize_results(rewards_train, steps_train, q, "8x8 FrozenLake (No Gym)")

    print("Running 10 rendered episodes with learned Q-table...")
    test(10, q, is_8x8=True, is_slippery=False, render=True, render_mode="human")

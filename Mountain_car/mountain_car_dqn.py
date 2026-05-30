import argparse
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit


@dataclass(frozen=True)
class DQNConfig:
    env_id: str = "MountainCar-v0"
    seed: int = 0

    # MountainCar-v0 default is 200. Keep 200 so returns are comparable (best is closer to 0).
    max_episode_steps: int = 200

    train_steps: int = 600_000

    # Discretization (turn continuous state into a discrete grid)
    pos_bins: int = 32
    vel_bins: int = 32

    # Tabular Q-learning
    alpha: float = 0.12
    alpha_end: float = 0.02
    alpha_decay_steps: int = 400_000

    # Eligibility traces (SARSA(λ))
    lambda_trace: float = 0.95
    goal_bonus: float = 200.0

    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 400_000

    # Potential-based shaping (keeps optimal policy, speeds learning)
    shaping_pos_k: float = 20.0
    shaping_vel_k: float = 5.0

    eval_every_steps: int = 20_000
    eval_episodes: int = 20


class Discretizer:
    def __init__(self, obs_low: np.ndarray, obs_high: np.ndarray, pos_bins: int, vel_bins: int):
        self.obs_low = obs_low.astype(np.float64)
        self.obs_high = obs_high.astype(np.float64)
        self.pos_bins = int(pos_bins)
        self.vel_bins = int(vel_bins)

        # Create bin edges (excluding endpoints).
        self.pos_edges = np.linspace(self.obs_low[0], self.obs_high[0], self.pos_bins + 1)[1:-1]
        self.vel_edges = np.linspace(self.obs_low[1], self.obs_high[1], self.vel_bins + 1)[1:-1]

    def obs_to_index(self, obs: np.ndarray) -> int:
        pos = float(obs[0])
        vel = float(obs[1])
        pos_i = int(np.digitize(pos, self.pos_edges))
        vel_i = int(np.digitize(vel, self.vel_edges))
        pos_i = int(np.clip(pos_i, 0, self.pos_bins - 1))
        vel_i = int(np.clip(vel_i, 0, self.vel_bins - 1))
        return pos_i * self.vel_bins + vel_i

    @property
    def n_states(self) -> int:
        return self.pos_bins * self.vel_bins


def _make_env(env_id: str, render_mode: str, max_episode_steps: int) -> gym.Env:
    rm = None if render_mode == "none" else render_mode
    # Gymnasium will wrap with TimeLimit to this max steps.
    return gym.make(env_id, render_mode=rm, max_episode_steps=max_episode_steps)


def _potential(obs: np.ndarray, k_pos: float, k_vel: float) -> float:
    # Potential-based shaping: Phi(s) = k_pos * position + k_vel * |velocity|
    return float(k_pos * obs[0] + k_vel * abs(obs[1]))


def evaluate(
    env_id: str,
    q_table: np.ndarray,
    discretizer: Discretizer,
    episodes: int,
    seed: int,
    max_episode_steps: int = 500,
    render_mode: str = "none",
    fps: float = 15.0,
    hold: bool = False,
) -> Tuple[float, float]:
    """Evaluate the Q-table policy (greedy)."""
    env = _make_env(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(seed)

    rgb_state = {"fig": None, "ax": None, "im": None}
    tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    step_counter = 0

    def _render() -> None:
        nonlocal step_counter
        if render_mode == "none":
            return

        step_counter += 1
        frame = env.render()

        if render_mode == "ansi" and isinstance(frame, str):
            if tty:
                print("\033[H\033[J", end="")
            print(f"step={step_counter}")
            print(frame)
            if fps and fps > 0:
                time.sleep(1.0 / fps)
            return

        if render_mode == "rgb_array" and isinstance(frame, np.ndarray):
            if rgb_state["fig"] is None:
                plt.ion()
                rgb_state["fig"], rgb_state["ax"] = plt.subplots(figsize=(6, 4))
                rgb_state["im"] = rgb_state["ax"].imshow(frame)
                rgb_state["ax"].axis("off")
            else:
                rgb_state["im"].set_data(frame)

            rgb_state["fig"].canvas.draw_idle()
            rgb_state["fig"].canvas.flush_events()
            if fps and fps > 0:
                plt.pause(1.0 / fps)
            return

        if fps and fps > 0:
            time.sleep(1.0 / fps)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        _render()

        while not terminated and not truncated:
            s = discretizer.obs_to_index(obs)
            action = int(np.argmax(q_table[s]))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            _render()

        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)

    if hold and render_mode != "none":
        if render_mode == "rgb_array" and rgb_state["fig"] is not None:
            plt.ioff()
            plt.show(block=True)
        else:
            try:
                input("\nDemo finished. Press Enter to close... ")
            except EOFError:
                time.sleep(2.0)

    env.close()
    if rgb_state["fig"] is not None:
        plt.close(rgb_state["fig"])

    return float(np.mean(episode_returns)), float(np.mean(episode_lengths))


def train(cfg: DQNConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Train discretized tabular SARSA(λ) on MountainCar (discrete actions)."""
    # Key trick: let training episodes run longer so the agent can discover
    # the momentum-building strategy (often requires "wasting" steps early on).
    # We still *evaluate/demo* with cfg.max_episode_steps (default 200).
    train_max_steps = int(max(cfg.max_episode_steps, 500))
    env = _make_env(cfg.env_id, render_mode="none", max_episode_steps=train_max_steps)
    obs, _ = env.reset(seed=cfg.seed)

    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    action_dim = int(env.action_space.n)

    rng = np.random.default_rng(cfg.seed)

    discretizer = Discretizer(obs_low, obs_high, pos_bins=cfg.pos_bins, vel_bins=cfg.vel_bins)
    q_table = np.zeros((discretizer.n_states, action_dim), dtype=np.float64)
    e_table = np.zeros_like(q_table)

    stats: Dict[str, List[float]] = {
        "train/episode_return": [],
        "train/episode_len": [],
        "eval/mean_return": [],
        "eval/mean_len": [],
        "update/epsilon": [],
        "update/alpha": [],
        "train/success": [],
    }

    steps_done = 0
    last_eval_steps = 0
    ep_return = 0.0
    ep_len = 0
    successes = 0

    def _select_action(state_index: int, epsilon: float) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, action_dim))
        return int(np.argmax(q_table[state_index]))

    while steps_done < cfg.train_steps:
        # Epsilon-greedy exploration
        frac = min(1.0, steps_done / max(1, cfg.epsilon_decay_steps))
        epsilon = float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))

        frac_a = min(1.0, steps_done / max(1, cfg.alpha_decay_steps))
        alpha = float(cfg.alpha + frac_a * (cfg.alpha_end - cfg.alpha))

        # Choose action (behavior policy)
        s = discretizer.obs_to_index(obs)
        action = _select_action(s, epsilon)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # Potential-based shaping: r' = r + gamma * Phi(s') - Phi(s)
        phi_s = _potential(obs, cfg.shaping_pos_k, cfg.shaping_vel_k)
        phi_sp = _potential(next_obs, cfg.shaping_pos_k, cfg.shaping_vel_k)
        shaped_reward = float(reward) + cfg.gamma * phi_sp - phi_s
        if terminated:
            shaped_reward += float(cfg.goal_bonus)

        ep_return += float(reward)  # Track original reward for stats
        ep_len += 1
        steps_done += 1

        # SARSA(λ) update (tabular with eligibility traces)
        sp = discretizer.obs_to_index(next_obs)
        if done:
            next_action = 0
            td_target = shaped_reward
        else:
            next_action = _select_action(sp, epsilon)
            td_target = shaped_reward + cfg.gamma * float(q_table[sp, next_action])

        td_error = td_target - float(q_table[s, action])

        # Replacing traces
        e_table *= cfg.gamma * cfg.lambda_trace
        e_table[s, action] = 1.0

        q_table += alpha * td_error * e_table

        if done:
            e_table.fill(0.0)

        stats["update/epsilon"].append(float(epsilon))
        stats["update/alpha"].append(float(alpha))

        # Periodic evaluation
        if steps_done - last_eval_steps >= cfg.eval_every_steps:
            mean_ret, mean_len = evaluate(
                cfg.env_id,
                q_table,
                discretizer,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + steps_done,
                max_episode_steps=cfg.max_episode_steps,
                render_mode="none",
            )
            stats["eval/mean_return"].append(mean_ret)
            stats["eval/mean_len"].append(mean_len)
            success_rate = successes / max(1, len(stats["train/episode_return"]))
            print(
                f"steps={steps_done:>7d} | eval_return={mean_ret:>7.2f} | eval_len={mean_len:>6.1f} | "
                f"eps={epsilon:.3f} | alpha={alpha:.3f} | success_rate={success_rate:.3f}"
            )
            last_eval_steps = steps_done

        # Episode termination
        if done:
            stats["train/episode_return"].append(ep_return)
            stats["train/episode_len"].append(float(ep_len))
            if terminated:
                successes += 1
                stats["train/success"].append(1.0)
            else:
                stats["train/success"].append(0.0)
            ep_return = 0.0
            ep_len = 0
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        else:
            obs = next_obs

    env.close()
    params = {
        "q_table": q_table,
        "pos_edges": discretizer.pos_edges,
        "vel_edges": discretizer.vel_edges,
        "pos_bins": np.array([discretizer.pos_bins], dtype=np.int64),
        "vel_bins": np.array([discretizer.vel_bins], dtype=np.int64),
        "obs_low": discretizer.obs_low,
        "obs_high": discretizer.obs_high,
    }
    return params, stats


def visualize_results(stats: Dict[str, List[float]], out_png: Path) -> None:
    """Visualize training results."""
    returns = np.array(stats.get("train/episode_return", []), dtype=np.float64)
    lens = np.array(stats.get("train/episode_len", []), dtype=np.float64)
    successes = np.array(stats.get("train/success", []), dtype=np.float64)

    if len(returns) == 0:
        print("No episode stats collected (unexpected).")
        return

    window = min(50, max(5, len(returns) // 10))
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(returns, alpha=0.3, color="steelblue", label="episode return")
    plt.plot(np.arange(window - 1, window - 1 + len(smooth)), smooth, color="navy", label=f"{window}-ep mean")
    plt.title("MountainCar (Tabular SARSA(λ)) — Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(successes) > 0:
        window_s = min(50, max(5, len(successes) // 10))
        succ_smooth = np.convolve(successes, np.ones(window_s) / window_s, mode="valid")
        plt.plot(np.arange(window_s - 1, window_s - 1 + len(succ_smooth)), succ_smooth, color="darkgreen", label="success rate")
        plt.title("Goal Reach Rate")
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Saved plot to: {out_png}")
    print(f"Final mean return (last {window} eps): {float(np.mean(returns[-window:])):.2f}")
    print(f"Final mean length (last {window} eps): {float(np.mean(lens[-window:])):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discretized tabular SARSA(λ) for Gymnasium MountainCar-v0 (NumPy, discrete actions)"
    )
    parser.add_argument("--env-id", type=str, default="MountainCar-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=600_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every-steps", type=int, default=20_000)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["none", "human", "ansi", "rgb_array"],
        help="Gymnasium render_mode to use during the post-training demo",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Demo playback speed")
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument(
        "--hold",
        action="store_true",
        help="After the demo, keep the render open until you press Enter",
    )
    args = parser.parse_args()

    cfg = DQNConfig(
        env_id=args.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        eval_every_steps=args.eval_every_steps,
    )

    print(
        f"Training discretized tabular SARSA(λ) on {cfg.env_id} | steps={cfg.train_steps} | "
        f"bins=({cfg.pos_bins}x{cfg.vel_bins}) | seed={cfg.seed} | "
        f"eval_max_steps={cfg.max_episode_steps}"
    )
    params, stats = train(cfg)

    out_dir = Path(__file__).resolve().parent
    model_path = out_dir / "mountain_car_dqn.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"config": cfg, "params": params, "stats": stats}, f)
    print(f"Saved model to: {model_path}")

    plot_path = out_dir / "mountain_car_dqn_result.png"
    visualize_results(stats, plot_path)

    q_table = params["q_table"]
    discretizer = Discretizer(params["obs_low"], params["obs_high"], int(params["pos_bins"][0]), int(params["vel_bins"][0]))
    discretizer.pos_edges = params["pos_edges"]
    discretizer.vel_edges = params["vel_edges"]

    mean_ret, mean_len = evaluate(
        cfg.env_id,
        q_table,
        discretizer,
        episodes=100,
        seed=cfg.seed + 999,
        max_episode_steps=cfg.max_episode_steps,
        render_mode="none",
    )
    print(f"\nEval (greedy policy) over 100 eps | mean_return={mean_ret:.2f} | mean_len={mean_len:.1f}")

    demo_render_mode = args.render_mode

    if demo_render_mode == "none":
        script = Path(__file__).name
        print(
            "\n(No rendering was requested.)\n"
            "To render like Gymnasium examples, re-run with one of:\n"
            f"  - ./.venv/bin/python Mountain_Car/{script} --train-steps {cfg.train_steps} --render-mode human --demo-episodes 1 --hold\n"
            f"  - ./.venv/bin/python Mountain_Car/{script} --train-steps {cfg.train_steps} --render-mode ansi --demo-episodes 1\n"
            "\nTip: `human` uses Gymnasium's pixel rendering; `ansi` renders in the terminal.\n"
            "If the window flashes and closes, add `--hold` (keeps it open)."
        )
    else:
        print(
            f"\nRunning demo (greedy policy) | render_mode={demo_render_mode} | "
            f"episodes={args.demo_episodes} | fps={args.fps}..."
        )
        evaluate(
            cfg.env_id,
            q_table,
            discretizer,
            episodes=args.demo_episodes,
            seed=cfg.seed + 2024,
            max_episode_steps=cfg.max_episode_steps,
            render_mode=demo_render_mode,
            fps=args.fps,
            hold=args.hold,
        )


if __name__ == "__main__":
    main()

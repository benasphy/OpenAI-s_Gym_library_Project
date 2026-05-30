import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ARSConfig:
    env_id: str = "MountainCarContinuous-v0"
    seed: int = 0

    max_episode_steps: int = 999

    train_iterations: int = 300
    directions: int = 32
    top_directions: int = 16

    step_size: float = 0.02
    noise_std: float = 0.03

    # Observation normalization
    obs_clip: float = 5.0

    eval_every: int = 10
    eval_episodes: int = 20

    # Early stop once it reliably solves
    early_stop: bool = True
    early_stop_success: float = 0.90
    early_stop_return: float = 80.0
    early_stop_patience: int = 3


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = float(tot_count)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class LinearPolicy:
    """Simple linear policy with tanh squashing: a = tanh(W x + b)"""

    def __init__(self, obs_dim: int, act_dim: int, rng: np.random.Generator):
        self.W = rng.normal(scale=0.1, size=(act_dim, obs_dim)).astype(np.float64)
        self.b = np.zeros((act_dim,), dtype=np.float64)

    def act(self, obs: np.ndarray) -> np.ndarray:
        z = self.W @ obs + self.b
        return np.tanh(z)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.reshape(-1), self.b.reshape(-1)])

    def set_params(self, p: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64)
        w_size = self.W.size
        self.W[...] = p[:w_size].reshape(self.W.shape)
        self.b[...] = p[w_size : w_size + self.b.size]

    @property
    def param_dim(self) -> int:
        return int(self.W.size + self.b.size)


def _make_env(env_id: str, render_mode: str, max_episode_steps: int) -> gym.Env:
    rm = None if render_mode == "none" else render_mode
    return gym.make(env_id, render_mode=rm, max_episode_steps=max_episode_steps)


def _rollout(
    env: gym.Env,
    policy_params: np.ndarray,
    policy_template: LinearPolicy,
    rms: RunningMeanStd,
    obs_clip: float,
    seed: int,
    render_mode: str = "none",
    fps: float = 30.0,
    update_rms: bool = True,
) -> Tuple[float, int, bool]:
    # clone template policy (cheap)
    policy = LinearPolicy(policy_template.W.shape[1], policy_template.W.shape[0], np.random.default_rng(0))
    policy.set_params(policy_params)

    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    ep_ret = 0.0
    ep_len = 0

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
        else:
            if fps and fps > 0:
                time.sleep(1.0 / fps)

    _render()

    obs_batch: List[np.ndarray] = []

    while not terminated and not truncated:
        obs_batch.append(np.asarray(obs, dtype=np.float64))

        # normalize obs
        norm_obs = (obs - rms.mean) / rms.std
        norm_obs = np.clip(norm_obs, -obs_clip, obs_clip)

        a = policy.act(norm_obs)
        # scale to env action bounds
        a = np.clip(a, -1.0, 1.0)
        if isinstance(env.action_space, gym.spaces.Box):
            a = a * env.action_space.high

        obs, reward, terminated, truncated, _ = env.step(a)
        ep_ret += float(reward)
        ep_len += 1

        _render()

    # Update running stats from this rollout (only from raw obs).
    # During evaluation/demo we freeze normalization for consistent reporting.
    if update_rms and len(obs_batch) > 0:
        rms.update(np.stack(obs_batch, axis=0))

    success = bool(terminated)
    return ep_ret, ep_len, success


def evaluate(
    cfg: ARSConfig,
    policy_params: np.ndarray,
    policy_template: LinearPolicy,
    rms: RunningMeanStd,
    episodes: int,
    seed: int,
    render_mode: str = "none",
    fps: float = 30.0,
    hold: bool = False,
) -> Tuple[float, float, float]:
    env = _make_env(cfg.env_id, render_mode=render_mode, max_episode_steps=cfg.max_episode_steps)
    rng = np.random.default_rng(seed)

    rets: List[float] = []
    lens: List[int] = []
    succ: int = 0

    for _ in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        r, l, s = _rollout(
            env,
            policy_params,
            policy_template,
            rms,
            obs_clip=cfg.obs_clip,
            seed=ep_seed,
            render_mode=render_mode,
            fps=fps,
            update_rms=False,
        )
        rets.append(r)
        lens.append(l)
        succ += int(s)

    if hold and render_mode != "none":
        try:
            input("\nDemo finished. Press Enter to close... ")
        except EOFError:
            time.sleep(2.0)

    env.close()
    return float(np.mean(rets)), float(np.mean(lens)), float(succ / max(1, episodes))


def train(cfg: ARSConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    env = _make_env(cfg.env_id, render_mode="none", max_episode_steps=cfg.max_episode_steps)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    rng = np.random.default_rng(cfg.seed)

    rms = RunningMeanStd((obs_dim,))
    policy_template = LinearPolicy(obs_dim, act_dim, rng)

    # Heuristic init for MountainCarContinuous: push in the direction of velocity.
    # This immediately creates the back-and-forth momentum pattern needed to reach the goal.
    if cfg.env_id == "MountainCarContinuous-v0" and act_dim == 1 and obs_dim >= 2:
        policy_template.W.fill(0.0)
        policy_template.b.fill(0.0)
        policy_template.W[0, 1] = 5.0

    theta = policy_template.get_params()

    stats: Dict[str, List[float]] = {
        "train/iter": [],
        "eval/mean_return": [],
        "eval/mean_len": [],
        "eval/success_rate": [],
    }

    best_theta = theta.copy()
    best_eval = -1e9

    solved_streak = 0

    for it in range(1, cfg.train_iterations + 1):
        deltas = rng.normal(size=(cfg.directions, theta.size)).astype(np.float64)

        rewards_pos = np.zeros((cfg.directions,), dtype=np.float64)
        rewards_neg = np.zeros((cfg.directions,), dtype=np.float64)

        # Evaluate positive/negative perturbations
        for k in range(cfg.directions):
            theta_pos = theta + cfg.noise_std * deltas[k]
            theta_neg = theta - cfg.noise_std * deltas[k]

            r_pos, _, _ = _rollout(
                env,
                theta_pos,
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=cfg.seed + it * 1000 + k * 2,
                update_rms=True,
            )
            r_neg, _, _ = _rollout(
                env,
                theta_neg,
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=cfg.seed + it * 1000 + k * 2 + 1,
                update_rms=True,
            )
            rewards_pos[k] = r_pos
            rewards_neg[k] = r_neg

        # Rank directions by max reward
        scores = np.maximum(rewards_pos, rewards_neg)
        top_k = int(min(cfg.top_directions, cfg.directions))
        top_idx = np.argsort(scores)[-top_k:]

        # Normalize by reward std for stable updates
        sigma_r = float(np.std(np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]])))
        if sigma_r < 1e-8:
            sigma_r = 1.0

        step = np.zeros_like(theta)
        for idx in top_idx:
            step += (rewards_pos[idx] - rewards_neg[idx]) * deltas[idx]

        theta = theta + (cfg.step_size / (top_k * sigma_r)) * step

        # Periodic evaluation
        if (it % cfg.eval_every) == 0 or it == 1 or it == cfg.train_iterations:
            mean_ret, mean_len, succ = evaluate(
                cfg,
                theta,
                policy_template,
                rms,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + 9999 + it,
                render_mode="none",
            )

            stats["train/iter"].append(float(it))
            stats["eval/mean_return"].append(float(mean_ret))
            stats["eval/mean_len"].append(float(mean_len))
            stats["eval/success_rate"].append(float(succ))

            if mean_ret > best_eval:
                best_eval = mean_ret
                best_theta = theta.copy()

            print(
                f"iter={it:>4d}/{cfg.train_iterations} | eval_return={mean_ret:>8.2f} | "
                f"eval_len={mean_len:>6.1f} | success_rate={succ:.3f} | best_return={best_eval:>8.2f}"
            )

            if cfg.early_stop and succ >= cfg.early_stop_success and mean_ret >= cfg.early_stop_return:
                solved_streak += 1
            else:
                solved_streak = 0

            if cfg.early_stop and solved_streak >= int(max(1, cfg.early_stop_patience)):
                print(
                    f"Early stop: solved for {solved_streak} evals "
                    f"(success>={cfg.early_stop_success:.2f}, return>={cfg.early_stop_return:.1f})."
                )
                break

    env.close()

    params = {
        "theta": best_theta,
        "obs_mean": rms.mean,
        "obs_std": rms.std,
        "obs_clip": np.array([cfg.obs_clip], dtype=np.float64),
        "obs_dim": np.array([obs_dim], dtype=np.int64),
        "act_dim": np.array([act_dim], dtype=np.int64),
    }
    return params, stats


def visualize_results(stats: Dict[str, List[float]], out_png: Path) -> None:
    iters = np.array(stats.get("train/iter", []), dtype=np.float64)
    rets = np.array(stats.get("eval/mean_return", []), dtype=np.float64)
    lens = np.array(stats.get("eval/mean_len", []), dtype=np.float64)
    succ = np.array(stats.get("eval/success_rate", []), dtype=np.float64)

    if len(iters) == 0:
        print("No eval stats collected (unexpected).")
        return

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iters, rets, color="navy")
    plt.title("MountainCarContinuous (ARS) — Eval Return")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(iters, succ, color="darkgreen", label="success rate")
    plt.plot(iters, lens / max(1.0, float(np.max(lens))), color="gray", alpha=0.5, label="len (scaled)")
    plt.title("Goal Reach Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous-action MountainCar (MountainCarContinuous-v0) solved via ARS policy search (NumPy)"
    )
    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=999)

    parser.add_argument("--train-iterations", type=int, default=300)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--top-directions", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.03)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)

    parser.add_argument(
        "--load-path",
        type=str,
        default="",
        help="Optional: load a previously saved .pkl and skip training (fast demo).",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training. Use with --load-path.",
    )

    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["none", "human", "ansi", "rgb_array"],
        help="Gymnasium render_mode to use during the post-training demo",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Demo playback speed")
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument(
        "--demo-until-success",
        action="store_true",
        help="Retry demo episodes until a success is observed (up to --demo-max-attempts).",
    )
    parser.add_argument(
        "--demo-max-attempts",
        type=int,
        default=20,
        help="Max demo rollouts when using --demo-until-success.",
    )

    args = parser.parse_args()

    cfg = ARSConfig(
        env_id=args.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        train_iterations=args.train_iterations,
        directions=args.directions,
        top_directions=args.top_directions,
        step_size=args.step_size,
        noise_std=args.noise_std,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )

    if cfg.env_id == "MountainCarContinuous-v0" and cfg.max_episode_steps < 500:
        print(
            f"WARNING: max_episode_steps={cfg.max_episode_steps} is usually too small for {cfg.env_id}. "
            "This task typically needs ~500-800 steps to build momentum. "
            "Use --max-episode-steps 999 for normal behavior."
        )

    out_dir = Path(__file__).resolve().parent
    default_model_path = out_dir / "mountain_car_continuous_ars.pkl"

    load_path = Path(args.load_path).expanduser() if args.load_path else None
    if args.no_train:
        if load_path is None:
            raise SystemExit("--no-train requires --load-path")
        if not load_path.exists():
            raise SystemExit(f"Load path not found: {load_path}")

        with open(load_path, "rb") as f:
            blob = pickle.load(f)
        params = blob["params"]
        stats = blob.get("stats", {})
        print(f"Loaded model from: {load_path}")
    else:
        print(
            f"Training ARS on {cfg.env_id} | iters={cfg.train_iterations} | dirs={cfg.directions} | "
            f"top={cfg.top_directions} | step_size={cfg.step_size} | noise_std={cfg.noise_std}"
        )
        params, stats = train(cfg)

    model_path = default_model_path
    if not args.no_train:
        with open(model_path, "wb") as f:
            pickle.dump({"config": cfg, "params": params, "stats": stats}, f)
        print(f"Saved model to: {model_path}")

    if not args.no_train:
        plot_path = out_dir / "mountain_car_continuous_ars_result.png"
        visualize_results(stats, plot_path)

    # Demo with best params
    obs_dim = int(params["obs_dim"][0])
    act_dim = int(params["act_dim"][0])
    policy_template = LinearPolicy(obs_dim, act_dim, np.random.default_rng(0))

    # Use frozen RMS for demo
    rms = RunningMeanStd((obs_dim,))
    rms.mean = np.asarray(params["obs_mean"], dtype=np.float64).copy()
    rms.var = np.asarray(params["obs_std"], dtype=np.float64).copy() ** 2

    mean_ret, mean_len, succ = evaluate(
        cfg,
        params["theta"],
        policy_template,
        rms,
        episodes=100,
        seed=cfg.seed + 2026,
        render_mode="none",
    )
    print(f"\nEval (best policy) over 100 eps | mean_return={mean_ret:.2f} | mean_len={mean_len:.1f} | success_rate={succ:.3f}")

    if args.render_mode == "none":
        script = Path(__file__).name
        print(
            "\n(No rendering was requested.)\n"
            "To render like Gymnasium examples, re-run with one of:\n"
            f"  - ./.venv/bin/python Mountain_Car/{script} --render-mode human --demo-episodes 1 --hold\n"
            f"  - ./.venv/bin/python Mountain_Car/{script} --render-mode ansi --demo-episodes 1\n"
        )
        return

    print(
        f"\nRunning demo (best policy) | render_mode={args.render_mode} | episodes={args.demo_episodes} | fps={args.fps}..."
    )

    # Demo uses its own env so rendering works
    demo_env = _make_env(cfg.env_id, render_mode=args.render_mode, max_episode_steps=cfg.max_episode_steps)
    rng = np.random.default_rng(cfg.seed + 777)

    target_episodes = int(max(1, args.demo_episodes))
    if not args.demo_until_success:
        for _ in range(target_episodes):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            _rollout(
                demo_env,
                params["theta"],
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=ep_seed,
                render_mode=args.render_mode,
                fps=args.fps,
                update_rms=False,
            )
    else:
        successes = 0
        attempts = 0
        max_attempts = int(max(target_episodes, args.demo_max_attempts))
        while successes < target_episodes and attempts < max_attempts:
            attempts += 1
            ep_seed = int(rng.integers(0, 2**31 - 1))
            _, _, success = _rollout(
                demo_env,
                params["theta"],
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=ep_seed,
                render_mode=args.render_mode,
                fps=args.fps,
                update_rms=False,
            )
            if success:
                successes += 1

        if successes < target_episodes:
            print(
                f"Demo note: only {successes}/{target_episodes} successes after {attempts} attempts. "
                "Increase --max-episode-steps or retrain for higher success_rate."
            )

    if args.hold:
        try:
            input("\nDemo finished. Press Enter to close... ")
        except EOFError:
            time.sleep(2.0)

    demo_env.close()


if __name__ == "__main__":
    main()

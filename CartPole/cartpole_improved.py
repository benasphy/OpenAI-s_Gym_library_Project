"""

    - For render_mode=human, this script does NOT sleep by default (max speed).
    - If you want slower playback, pass --throttle-fps 60 (or 30).
"""

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ARSConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    max_episode_steps: int = 500

    train_iterations: int = 150
    directions: int = 32
    top_directions: int = 16
    step_size: float = 0.02
    noise_std: float = 0.03

    obs_clip: float = 5.0

    eval_every: int = 10
    eval_episodes: int = 50

    early_stop: bool = True
    early_stop_success: float = 0.90
    early_stop_return: float = 475.0
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


class LinearDiscretePolicy:
    """Linear policy for discrete actions: a = argmax(W x + b)."""

    def __init__(self, obs_dim: int, act_dim: int, rng: np.random.Generator):
        self.W = rng.normal(scale=0.1, size=(act_dim, obs_dim)).astype(np.float64)
        self.b = np.zeros((act_dim,), dtype=np.float64)

    def act(self, obs: np.ndarray) -> int:
        logits = self.W @ obs + self.b
        return int(np.argmax(logits))

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


def _normalize_obs(obs: np.ndarray, rms: RunningMeanStd, obs_clip: float) -> np.ndarray:
    z = (obs - rms.mean) / rms.std
    return np.clip(z, -obs_clip, obs_clip)


def _rollout(
    env: gym.Env,
    theta: np.ndarray,
    policy_template: LinearDiscretePolicy,
    rms: RunningMeanStd,
    obs_clip: float,
    seed: int,
    render_mode: str = "none",
    fps: float = 60.0,
    throttle_fps: float = 0.0,
    update_rms: bool = True,
) -> Tuple[float, int, bool]:
    policy = LinearDiscretePolicy(policy_template.W.shape[1], policy_template.W.shape[0], np.random.default_rng(0))
    policy.set_params(theta)

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
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
        if render_mode == "rgb_array" and frame is not None:
            # Gymnasium can return an array here; we don't display it inline.
            pass
        elif render_mode == "human":
            # Gymnasium viewer updates automatically; do not sleep unless user asked.
            pass

        # Throttle only if explicitly requested (useful for slower visual playback)
        if throttle_fps and throttle_fps > 0:
            time.sleep(1.0 / float(throttle_fps))

    while not (terminated or truncated):
        if update_rms:
            rms.update(obs[None, :])
        obs_n = _normalize_obs(obs, rms, obs_clip)
        action = policy.act(obs_n)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float64)
        ep_ret += float(reward)
        ep_len += 1

        if tty and render_mode == "rgb_array":
            print(f"step={step_counter} ret={ep_ret:.1f} len={ep_len}")
        _render()

    success = (not terminated) and bool(truncated)
    return ep_ret, ep_len, success


def evaluate(
    cfg: ARSConfig,
    theta: np.ndarray,
    policy_template: LinearDiscretePolicy,
    rms: RunningMeanStd,
    episodes: int,
    seed: int,
    render_mode: str = "none",
    fps: float = 60.0,
    throttle_fps: float = 0.0,
) -> Tuple[float, float, float]:
    env = _make_env(cfg.env_id, render_mode=render_mode, max_episode_steps=cfg.max_episode_steps)
    rets: List[float] = []
    lens: List[int] = []
    succ: List[float] = []
    for i in range(int(episodes)):
        ep_ret, ep_len, ok = _rollout(
            env,
            theta,
            policy_template,
            rms,
            obs_clip=cfg.obs_clip,
            seed=seed + i,
            render_mode=render_mode,
            fps=fps,
            throttle_fps=throttle_fps,
            update_rms=False,
        )
        rets.append(ep_ret)
        lens.append(ep_len)
        succ.append(1.0 if ok else 0.0)
    env.close()
    return float(np.mean(rets)), float(np.mean(lens)), float(np.mean(succ))


def train(cfg: ARSConfig) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    env = _make_env(cfg.env_id, render_mode="none", max_episode_steps=cfg.max_episode_steps)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    rng = np.random.default_rng(cfg.seed)
    policy_template = LinearDiscretePolicy(obs_dim, act_dim, rng)
    rms = RunningMeanStd((obs_dim,))

    theta = policy_template.get_params().copy()
    best_theta = theta.copy()

    stats: Dict[str, List[float]] = {
        "iter": [],
        "eval_mean_return": [],
        "eval_success_rate": [],
    }

    patience = 0

    for it in range(1, int(cfg.train_iterations) + 1):
        deltas = rng.normal(size=(cfg.directions, theta.size)).astype(np.float64)
        pos_rets = np.zeros((cfg.directions,), dtype=np.float64)
        neg_rets = np.zeros((cfg.directions,), dtype=np.float64)

        for k in range(cfg.directions):
            seed_base = cfg.seed * 1_000_000 + it * 10_000 + k
            pos_theta = theta + cfg.noise_std * deltas[k]
            neg_theta = theta - cfg.noise_std * deltas[k]
            pos_rets[k], _, _ = _rollout(
                env,
                pos_theta,
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=seed_base,
                render_mode="none",
                update_rms=True,
            )
            neg_rets[k], _, _ = _rollout(
                env,
                neg_theta,
                policy_template,
                rms,
                obs_clip=cfg.obs_clip,
                seed=seed_base + 1,
                render_mode="none",
                update_rms=True,
            )

        scores = np.maximum(pos_rets, neg_rets)
        top_idx = np.argsort(scores)[-int(cfg.top_directions) :]

        reward_std = float(np.std(np.concatenate([pos_rets[top_idx], neg_rets[top_idx]])))
        if reward_std < 1e-8:
            reward_std = 1.0

        step = np.zeros_like(theta)
        for idx in top_idx:
            step += (pos_rets[idx] - neg_rets[idx]) * deltas[idx]
        step /= (int(cfg.top_directions) * reward_std)
        theta = theta + cfg.step_size * step

        if it % int(cfg.eval_every) == 0 or it == cfg.train_iterations:
            mean_ret, mean_len, succ = evaluate(
                cfg,
                theta,
                policy_template,
                rms,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + 2026,
                render_mode="none",
            )
            stats["iter"].append(float(it))
            stats["eval_mean_return"].append(mean_ret)
            stats["eval_success_rate"].append(succ)

            if len(stats["eval_mean_return"]) == 1 or mean_ret >= max(stats["eval_mean_return"]):
                best_theta = theta.copy()

            print(
                f"iter={it:4d} | eval_mean_return={mean_ret:7.2f} | eval_mean_len={mean_len:6.1f} | success_rate={succ:.3f}"
            )

            if cfg.early_stop and succ >= cfg.early_stop_success and mean_ret >= cfg.early_stop_return:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print("Early stop: solved reliably.")
                    break
            else:
                patience = 0

    env.close()

    # Package parameters + normalization stats
    blob = {
        "env_id": cfg.env_id,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "theta": best_theta,
        "obs_mean": rms.mean.copy(),
        "obs_std": rms.std.copy(),
    }
    return blob, stats


def _print_stress_report(returns: np.ndarray, lengths: np.ndarray, success_rate: float, elapsed_s: float) -> None:
    returns = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if returns.size == 0:
        print("No episodes were run.")
        return
    sps = float(lengths.sum() / max(1e-9, elapsed_s))
    p50 = float(np.percentile(returns, 50))
    p90 = float(np.percentile(returns, 90))
    p99 = float(np.percentile(returns, 99))
    print(
        "\nStress report"
        f"\n  episodes:      {returns.size}"
        f"\n  mean_return:   {float(returns.mean()):.2f}"
        f"\n  max_return:    {float(returns.max()):.2f}"
        f"\n  p50/p90/p99:   {p50:.2f} / {p90:.2f} / {p99:.2f}"
        f"\n  mean_length:   {float(lengths.mean()):.1f}"
        f"\n  max_length:    {float(lengths.max()):.1f}"
        f"\n  success_rate:  {float(success_rate):.3f}"
        f"\n  steps/sec:     {sps:,.0f}"
    )


def _run_stress(
    cfg: ARSConfig,
    theta: np.ndarray,
    policy_template: LinearDiscretePolicy,
    rms: RunningMeanStd,
    episodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    env = _make_env(cfg.env_id, render_mode="none", max_episode_steps=cfg.max_episode_steps)
    returns: List[float] = []
    lengths: List[int] = []
    succs: List[float] = []

    t0 = time.perf_counter()
    for i in range(int(max(1, episodes))):
        ep_ret, ep_len, ok = _rollout(
            env,
            theta,
            policy_template,
            rms,
            obs_clip=cfg.obs_clip,
            seed=seed + i,
            render_mode="none",
            update_rms=False,
        )
        returns.append(ep_ret)
        lengths.append(ep_len)
        succs.append(1.0 if ok else 0.0)
    elapsed = time.perf_counter() - t0
    env.close()

    return np.asarray(returns, dtype=np.float64), np.asarray(lengths, dtype=np.float64), float(np.mean(succs)), float(elapsed)


def _parse_int_list(csv: str) -> List[int]:
    items: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole-v1 solved via ARS (NumPy, Gymnasium)")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--train-iterations", type=int, default=150)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--top-directions", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.03)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)

    parser.add_argument(
        "--load-path",
        type=str,
        default="",
        help="Optional: load a previously saved .pkl and skip training (fast demo).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force training even if a saved model exists.",
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
        choices=["none", "human", "rgb_array"],
        help="Gymnasium render_mode to use during the demo",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="(legacy) Unused for human; kept for compatibility")
    parser.add_argument(
        "--throttle-fps",
        type=float,
        default=0.0,
        help="If >0, sleeps to limit demo speed (useful for human rendering).",
    )
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument("--hold", action="store_true")

    parser.add_argument("--stress", action="store_true", help="Run many episodes (no rendering) and print stats")
    parser.add_argument("--stress-episodes", type=int, default=500)

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Benchmark multiple max_episode_steps values (no rendering).",
    )
    parser.add_argument(
        "--sweep-steps",
        type=str,
        default="500,1000,5000,20000,100000",
        help="Comma-separated list of max_episode_steps values to test.",
    )
    parser.add_argument("--sweep-episodes", type=int, default=50)

    parser.add_argument(
        "--until-fail",
        action="store_true",
        help="Run a single episode until terminated/truncated and print the step where it ended.",
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

    out_dir = Path(__file__).resolve().parent
    default_model_path = out_dir / "cartpole_ars.pkl"

    load_path = Path(args.load_path).expanduser() if args.load_path else None
    # Choose model source:
    # 1) explicit --no-train + --load-path
    # 2) explicit --train (force)
    # 3) otherwise: auto-load default model if it exists; else train once
    if args.no_train:
        if load_path is None:
            raise SystemExit("--no-train requires --load-path")
        if not load_path.exists():
            raise SystemExit(f"Load path not found: {load_path}")
        chosen_path = load_path
        with open(chosen_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        stats = saved.get("stats", {})
        print(f"Loaded model from: {chosen_path}")
    elif (not args.train) and default_model_path.exists():
        chosen_path = default_model_path
        with open(chosen_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        stats = saved.get("stats", {})
        print(f"Loaded model from: {chosen_path}")
    else:
        print(
            f"Training ARS on {cfg.env_id} | iters={cfg.train_iterations} | dirs={cfg.directions} | top={cfg.top_directions} | "
            f"step_size={cfg.step_size} | noise_std={cfg.noise_std}"
        )
        model, stats = train(cfg)
        with open(default_model_path, "wb") as f:
            pickle.dump({"config": cfg, "model": model, "stats": stats}, f)
        print(f"Saved model to: {default_model_path}")

    obs_dim = int(model["obs_dim"])
    act_dim = int(model["act_dim"])
    policy_template = LinearDiscretePolicy(obs_dim, act_dim, np.random.default_rng(0))

    # Frozen normalization for eval/demo
    rms = RunningMeanStd((obs_dim,))
    rms.mean = np.asarray(model["obs_mean"], dtype=np.float64).copy()
    rms.var = np.asarray(model["obs_std"], dtype=np.float64).copy() ** 2

    mean_ret, mean_len, succ = evaluate(
        cfg,
        np.asarray(model["theta"], dtype=np.float64),
        policy_template,
        rms,
        episodes=100,
        seed=cfg.seed + 4040,
        render_mode="none",
    )
    print(f"\nEval over 100 eps | mean_return={mean_ret:.2f} | mean_len={mean_len:.1f} | success_rate={succ:.3f}")

    if args.stress:
        returns, lengths, success_rate, elapsed = _run_stress(
            cfg,
            np.asarray(model["theta"], dtype=np.float64),
            policy_template,
            rms,
            episodes=args.stress_episodes,
            seed=cfg.seed + 9000,
        )
        _print_stress_report(returns, lengths, success_rate, elapsed)
        return

    if args.sweep:
        steps_list = _parse_int_list(args.sweep_steps)
        print("\nSweep (no rendering)")
        print("  max_steps | mean_return | success | steps/sec")
        for ms in steps_list:
            cfg2 = ARSConfig(
                env_id=cfg.env_id,
                seed=cfg.seed,
                max_episode_steps=int(ms),
                train_iterations=cfg.train_iterations,
                directions=cfg.directions,
                top_directions=cfg.top_directions,
                step_size=cfg.step_size,
                noise_std=cfg.noise_std,
                obs_clip=cfg.obs_clip,
                eval_every=cfg.eval_every,
                eval_episodes=cfg.eval_episodes,
                early_stop=cfg.early_stop,
                early_stop_success=cfg.early_stop_success,
                early_stop_return=cfg.early_stop_return,
                early_stop_patience=cfg.early_stop_patience,
            )
            returns, lengths, success_rate, elapsed = _run_stress(
                cfg2,
                np.asarray(model["theta"], dtype=np.float64),
                policy_template,
                rms,
                episodes=args.sweep_episodes,
                seed=cfg.seed + 12000 + int(ms),
            )
            sps = float(lengths.sum() / max(1e-9, elapsed))
            print(
                f"  {ms:8d} | {float(returns.mean()):11.2f} | {success_rate:7.3f} | {sps:9.0f}"
            )
        return

    if args.until_fail:
        env = _make_env(cfg.env_id, render_mode=args.render_mode, max_episode_steps=cfg.max_episode_steps)
        theta = np.asarray(model["theta"], dtype=np.float64)
        obs, _ = env.reset(seed=cfg.seed + 555)
        obs = np.asarray(obs, dtype=np.float64)
        terminated = False
        truncated = False
        t0 = time.perf_counter()
        steps = 0
        ret = 0.0
        while not (terminated or truncated):
            obs_n = _normalize_obs(obs, rms, cfg.obs_clip)
            action = int(np.argmax(policy_template.W @ obs_n + policy_template.b))
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float64)
            ret += float(reward)
            steps += 1
            if args.render_mode != "none":
                env.render()
                if args.throttle_fps and args.throttle_fps > 0:
                    time.sleep(1.0 / float(args.throttle_fps))
        elapsed = time.perf_counter() - t0
        env.close()
        end_reason = "terminated" if terminated else "truncated"
        print(
            f"\nUntil-fail result: {end_reason} at step={steps} | return={ret:.1f} | seconds={elapsed:.2f} | steps/sec={steps/max(1e-9, elapsed):.0f}"
        )
        print(f"Final obs: {obs}")
        return

    if args.render_mode == "none":
        script = Path(__file__).name
        print(
            "\n(No rendering requested.)\n"
            "To render with Gymnasium's viewer, re-run with:\n"
            f"  - ./.venv/bin/python CartPole/{script} --render-mode human --demo-episodes 1 --hold\n"
        )
        return

    print(f"\nRunning demo | render_mode={args.render_mode} | episodes={args.demo_episodes} | fps={args.fps}...")
    demo_env = _make_env(cfg.env_id, render_mode=args.render_mode, max_episode_steps=cfg.max_episode_steps)
    rng = np.random.default_rng(cfg.seed + 777)
    for _ in range(int(max(1, args.demo_episodes))):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        _rollout(
            demo_env,
            np.asarray(model["theta"], dtype=np.float64),
            policy_template,
            rms,
            obs_clip=cfg.obs_clip,
            seed=ep_seed,
            render_mode=args.render_mode,
            fps=args.fps,
            throttle_fps=args.throttle_fps,
            update_rms=False,
        )

    if args.hold:
        try:
            input("\nDemo finished. Press Enter to close... ")
        except EOFError:
            time.sleep(2.0)
    demo_env.close()


if __name__ == "__main__":
    main()

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit


@dataclass(frozen=True)
class PPOConfig:
    env_id: str = "CliffWalking-v1"
    seed: int = 0

    max_episode_steps: int = 500

    train_steps: int = 300_000
    rollout_len: int = 1024

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    policy_lr: float = 0.05
    value_lr: float = 0.1

    epochs: int = 6
    minibatch_size: int = 256

    adv_norm: bool = True

    eval_every_updates: int = 20
    eval_episodes: int = 50


class Adam:
    def __init__(
        self,
        shape: Tuple[int, ...],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads * grads)

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / np.sum(exp)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    log_denom = np.log(np.sum(np.exp(z)))
    return z - log_denom


def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return z - log_denom


def _one_hot(n: int, idx: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[idx] = 1.0
    return v


def _policy_action_and_logp(rng: np.random.Generator, logits_row: np.ndarray) -> Tuple[int, float]:
    probs = _softmax(logits_row)
    action = int(rng.choice(len(probs), p=probs))
    logp = float(np.log(probs[action] + 1e-12))
    return action, logp


def _greedy_action(logits_row: np.ndarray, rng: np.random.Generator | None = None) -> int:
    max_val = np.max(logits_row)
    max_actions = np.flatnonzero(np.isclose(logits_row, max_val))
    if rng is None or len(max_actions) == 1:
        return int(max_actions[0])
    return int(rng.choice(max_actions))


def _gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (GAE-Lambda).

    dones[t] is 1.0 when episode ended after step t.
    """
    t_max = len(rewards)
    advantages = np.zeros(t_max, dtype=np.float64)
    last_gae = 0.0

    for t in reversed(range(t_max)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def evaluate(
    env_id: str,
    logits: np.ndarray,
    episodes: int,
    seed: int,
    max_episode_steps: int = 500,
    mode: str = "sample",
    render_mode: str = "none",
    fps: float = 15.0,
    render_every: int = 1,
    hold: bool = False,
) -> Tuple[float, float]:

    if render_mode == "none":
        gym_render_mode = None
    else:
        gym_render_mode = render_mode

    env = TimeLimit(gym.make(env_id, render_mode=gym_render_mode), max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(seed)

    rgb_state = {"fig": None, "ax": None, "im": None}
    tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    step_counter = 0

    def _render() -> None:
        nonlocal step_counter
        if render_mode == "none":
            return

        step_counter += 1
        if render_every > 1 and (step_counter % render_every) != 0:
            return

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

        # render_mode == "human" returns None and handles its own display
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
            if mode == "greedy":
                action = _greedy_action(logits[int(obs)], rng=rng)
            else:
                action, _ = _policy_action_and_logp(rng, logits[int(obs)])
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            _render()

        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)

    if hold and render_mode != "none":
        # Keep the process alive so Gymnasium's human viewer / matplotlib window stays open.
        # (Otherwise a solved episode can finish in ~17 steps and immediately exit.)
        if render_mode == "rgb_array" and rgb_state["fig"] is not None:
            plt.ioff()
            plt.show(block=True)
        else:
            try:
                input("\nDemo finished. Press Enter to close... ")
            except EOFError:
                # Non-interactive environment: just pause briefly.
                time.sleep(2.0)

    env.close()
    if rgb_state["fig"] is not None:
        plt.close(rgb_state["fig"])
    return float(np.mean(episode_returns)), float(np.mean(episode_lengths))


def _policy_grid(env: gym.Env, logits: np.ndarray) -> str:
    unwrapped = env.unwrapped
    nrow, ncol = unwrapped.shape
    cliff = getattr(unwrapped, "_cliff", None)
    # Gymnasium's CliffWalking keeps these on the unwrapped env, but fall back to the
    # canonical layout (start bottom-left, goal bottom-right) just in case.
    start_state = int(getattr(unwrapped, "start_state", (nrow - 1) * ncol))
    terminal_state = int(getattr(unwrapped, "terminal_state", (nrow - 1) * ncol + (ncol - 1)))

    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    lines = []
    for r in range(nrow):
        row_cells = []
        for c in range(ncol):
            s = r * ncol + c
            if s == start_state:
                row_cells.append("S")
                continue
            if s == terminal_state:
                row_cells.append("G")
                continue
            if cliff is not None and bool(cliff[r, c]):
                row_cells.append("X")
                continue

            a = _greedy_action(logits[s])
            row_cells.append(arrows.get(a, "?"))

        lines.append(" ".join(row_cells))

    return "\n".join(lines)


def train(cfg: PPOConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    env = TimeLimit(gym.make(cfg.env_id), max_episode_steps=cfg.max_episode_steps)
    obs, _ = env.reset(seed=cfg.seed)

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    rng = np.random.default_rng(cfg.seed)

    logits = rng.normal(loc=0.0, scale=0.01, size=(n_states, n_actions)).astype(np.float64)
    values_table = np.zeros((n_states,), dtype=np.float64)

    policy_opt = Adam(logits.shape, lr=cfg.policy_lr)
    value_opt = Adam(values_table.shape, lr=cfg.value_lr)

    stats: Dict[str, List[float]] = {
        "train/episode_return": [],
        "train/episode_len": [],
        "eval/mean_return": [],
        "eval/mean_len": [],
        "update/approx_kl": [],
        "update/clip_frac": [],
        "update/entropy": [],
    }

    steps_done = 0
    update_idx = 0

    ep_return = 0.0
    ep_len = 0

    while steps_done < cfg.train_steps:
        update_idx += 1

        states = np.zeros((cfg.rollout_len,), dtype=np.int32)
        actions = np.zeros((cfg.rollout_len,), dtype=np.int32)
        rewards = np.zeros((cfg.rollout_len,), dtype=np.float64)
        dones = np.zeros((cfg.rollout_len,), dtype=np.float64)
        old_logp = np.zeros((cfg.rollout_len,), dtype=np.float64)
        values = np.zeros((cfg.rollout_len,), dtype=np.float64)
        next_values = np.zeros((cfg.rollout_len,), dtype=np.float64)

        for t in range(cfg.rollout_len):
            s = int(obs)
            a, lp = _policy_action_and_logp(rng, logits[s])

            states[t] = s
            actions[t] = a
            old_logp[t] = lp
            values[t] = values_table[s]

            obs2, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)

            rewards[t] = float(r)
            dones[t] = 1.0 if done else 0.0

            ep_return += float(r)
            ep_len += 1

            if done:
                stats["train/episode_return"].append(ep_return)
                stats["train/episode_len"].append(float(ep_len))
                ep_return = 0.0
                ep_len = 0
                obs2, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

            next_values[t] = values_table[int(obs2)]
            obs = obs2

        steps_done += cfg.rollout_len

        advantages, returns = _gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        if cfg.adv_norm:
            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages) + 1e-8)
            advantages = (advantages - adv_mean) / adv_std

        # PPO optimization
        indices = np.arange(cfg.rollout_len)
        approx_kl_acc = 0.0
        clip_frac_acc = 0.0
        entropy_acc = 0.0
        mb_count = 0

        for _ in range(cfg.epochs):
            rng.shuffle(indices)
            for start in range(0, cfg.rollout_len, cfg.minibatch_size):
                mb_idx = indices[start : start + cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                mb_size = len(mb_idx)

                # Vectorized policy + value gradients (tabular => very fast)
                grad_logits = np.zeros_like(logits)
                grad_values = np.zeros_like(values_table)

                logp_all = _log_softmax_rows(logits)  # (S, A)
                probs_all = np.exp(logp_all)

                new_logp = logp_all[mb_states, mb_actions]
                ratio = np.exp(new_logp - mb_old_logp)
                clipped_ratio = np.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

                surr1 = ratio * mb_adv
                surr2 = clipped_ratio * mb_adv

                use_surr1 = surr1 < surr2
                in_clip = (ratio >= (1.0 - cfg.clip_eps)) & (ratio <= (1.0 + cfg.clip_eps))
                coef = np.where(use_surr1 | in_clip, ratio * mb_adv, 0.0)

                # Policy gradient: -(1/B) * coef * (onehot(a) - p)
                probs_rows = probs_all[mb_states]  # (B, A)
                scaled = (coef / mb_size).reshape(-1, 1)
                np.add.at(grad_logits, mb_states, scaled * probs_rows)
                np.add.at(grad_logits, (mb_states, mb_actions), -(coef / mb_size))

                # Entropy bonus gradient: loss includes -ent_coef * H
                # H(s) = -sum_a p log p ; dH/dz = p * (S - (logp + 1))
                H_all = -np.sum(probs_all * logp_all, axis=1)  # (S,)
                S_all = np.sum(probs_all * (logp_all + 1.0), axis=1)  # (S,)
                grad_H_all = probs_all * (S_all[:, None] - (logp_all + 1.0))
                np.add.at(grad_logits, mb_states, (-cfg.ent_coef / mb_size) * grad_H_all[mb_states])

                # Value function gradient: vf_coef * mean(0.5*(V-ret)^2)
                v_s = values_table[mb_states]
                np.add.at(grad_values, mb_states, (cfg.vf_coef / mb_size) * (v_s - mb_returns))

                # Diagnostics
                approx_kl = float(np.mean(mb_old_logp - new_logp))
                clip_frac = float(np.mean(np.abs(ratio - 1.0) > cfg.clip_eps))
                entropy = float(np.mean(H_all[mb_states]))

                # Apply updates
                logits = policy_opt.step(logits, grad_logits)
                values_table = value_opt.step(values_table, grad_values)

                approx_kl_acc += approx_kl
                clip_frac_acc += clip_frac
                entropy_acc += entropy
                mb_count += 1

        if mb_count > 0:
            stats["update/approx_kl"].append(float(approx_kl_acc / mb_count))
            stats["update/clip_frac"].append(float(clip_frac_acc / mb_count))
            stats["update/entropy"].append(float(entropy_acc / mb_count))

        if update_idx % cfg.eval_every_updates == 0:
            mean_ret, mean_len = evaluate(
                cfg.env_id,
                logits,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + update_idx,
                max_episode_steps=cfg.max_episode_steps,
                mode="sample",
            )
            stats["eval/mean_return"].append(mean_ret)
            stats["eval/mean_len"].append(mean_len)
            print(
                f"update={update_idx:04d} | steps={steps_done:>7d} | "
                f"eval_return={mean_ret:>7.2f} | eval_len={mean_len:>6.1f} | "
                f"kl={stats['update/approx_kl'][-1]:.4f} | clip={stats['update/clip_frac'][-1]:.3f} | "
                f"ent={stats['update/entropy'][-1]:.3f}"
            )

    env.close()
    params = {"logits": logits, "values": values_table, "env_id": np.array([cfg.env_id])}
    return params, stats


def visualize_results(stats: Dict[str, List[float]], out_png: Path) -> None:
    returns = np.array(stats.get("train/episode_return", []), dtype=np.float64)
    lens = np.array(stats.get("train/episode_len", []), dtype=np.float64)

    if len(returns) == 0:
        print("No episode stats collected (unexpected).")
        return

    window = min(200, len(returns))
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(returns, alpha=0.3, color="steelblue", label="episode return")
    plt.plot(np.arange(window - 1, window - 1 + len(smooth)), smooth, color="navy", label=f"{window}-ep mean")
    plt.title("CliffWalking PPO (tabular) — Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lens, alpha=0.4, color="darkred")
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Saved plot to: {out_png}")
    print(f"Final mean return (last {window} eps): {float(np.mean(returns[-window:])):.2f}")
    print(f"Final mean length (last {window} eps): {float(np.mean(lens[-window:])):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO for Gymnasium CliffWalking-v1 (NumPy, tabular)")
    parser.add_argument("--env-id", type=str, default="CliffWalking-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--rollout-len", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-every-updates", type=int, default=20)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["none", "human", "ansi", "rgb_array"],
        help="Gymnasium render_mode to use during the post-training demo",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Demo playback speed")
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N environment steps (useful for ansi / logging capture)",
    )
    parser.add_argument(
        "--demo-policy",
        type=str,
        default="sample",
        choices=["sample", "greedy"],
        help="Which policy to run in the demo",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Shortcut for --render-mode ansi",
    )
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument(
        "--auto-demo",
        action="store_true",
        help="Automatically run a short demo after training (requires --render/--render-mode)",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="After the demo, keep the render open until you press Enter",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        env_id=args.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        train_steps=args.train_steps,
        rollout_len=args.rollout_len,
        eval_episodes=args.eval_episodes,
        eval_every_updates=args.eval_every_updates,
    )

    print(f"Training PPO on {cfg.env_id} | steps={cfg.train_steps} | seed={cfg.seed}")
    params, stats = train(cfg)

    out_dir = Path(__file__).resolve().parent
    model_path = out_dir / "cliff_walking_ppo.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"config": cfg, "params": params, "stats": stats}, f)
    print(f"Saved model to: {model_path}")

    plot_path = out_dir / "cliff_walking_ppo_result.png"
    visualize_results(stats, plot_path)

    # Print a policy map
    env = gym.make(cfg.env_id)
    print("\nLearned greedy policy map (S=start, G=goal, X=cliff):")
    print(_policy_grid(env, params["logits"]))
    env.close()

    mean_ret, mean_len = evaluate(
        cfg.env_id,
        params["logits"],
        episodes=200,
        seed=cfg.seed + 999,
        max_episode_steps=cfg.max_episode_steps,
        mode="sample",
    )
    print(f"\nEval (stochastic policy) over 200 eps | mean_return={mean_ret:.2f} | mean_len={mean_len:.1f}")

    demo_render_mode = "ansi" if args.render and args.render_mode == "none" else args.render_mode

    if demo_render_mode == "none":
        script = Path(__file__).name
        print(
            "\n(No rendering was requested.)\n"
            "To render like Gymnasium examples, re-run with one of:\n"
            f"  - ./.venv/bin/python Cliff_Walking/{script} --train-steps {cfg.train_steps} --max-episode-steps {cfg.max_episode_steps} --render-mode ansi --demo-episodes 1\n"
            f"  - ./.venv/bin/python Cliff_Walking/{script} --train-steps {cfg.train_steps} --max-episode-steps {cfg.max_episode_steps} --render-mode human --demo-episodes 1\n"
            f"  - ./.venv/bin/python Cliff_Walking/{script} --train-steps {cfg.train_steps} --max-episode-steps {cfg.max_episode_steps} --render-mode rgb_array --demo-episodes 1\n"
            "\nTip: `ansi` renders in the terminal; `rgb_array` uses a matplotlib window.\n"
            "If the window flashes and closes, add `--hold` (keeps it open)."
        )
    else:
        # Demo is optional: only run if rendering is enabled.
        if args.auto_demo or demo_render_mode != "none":
            print(
                f"\nRunning demo ({args.demo_policy} policy) | render_mode={demo_render_mode} | "
                f"episodes={args.demo_episodes} | fps={args.fps}..."
            )
            evaluate(
                cfg.env_id,
                params["logits"],
                episodes=args.demo_episodes,
                seed=cfg.seed + 2024,
                max_episode_steps=cfg.max_episode_steps,
                mode=args.demo_policy,
                render_mode=demo_render_mode,
                fps=args.fps,
                render_every=args.render_every,
                hold=args.hold,
            )


if __name__ == "__main__":
    main()

"""Regenerate equivalent Lunar Lander episodes from the open Gymnasium environment.

The dataset was HANDED TO US (collected by the team; it was not generated in this project) and is not
redistributed here. The team reproduces the results exactly by pointing PIWM_DATA_ROOT at that dataset.
This script lets anyone else regenerate FUNCTIONALLY equivalent data (same format, same physics, a
similar pose distribution) from the open environment, so everything here can be re-run. Numbers will be
close but not bit-identical to the shipped figures, because the episodes differ.

How the originals were made is INFERRED, not known: our data characterization found an almost perfectly
UNIFORM distribution over the four discrete actions (plus a wide, uncontrolled pose spread) — the signature
of a RANDOM policy, since a real controller would bias toward the main engine. So the originals were very
likely random-action rollouts of Gymnasium LunarLander-v3, and `--policy random` (the default) is our best
reconstruction of the generating process, not a confirmed recipe.

Format produced per episode (matches the originals):
    <i>.npz  with  imgs (T,100,150,3) uint8,  acts (T,) int32,  states (T,8) float32

Requires Box2D:  pip install "gymnasium[box2d]"
Usage:
    python generate_data.py --n_train 345 --n_test 55 --out ./data/lunar
    python example_use.py        # ./data/lunar is the default data root; no env var needed
"""
import argparse
import os

import numpy as np

IMG_HW = (100, 150)   # (H, W); LunarLander renders 400x600, downscaled 4x


def heuristic(obs):
    """The classic discrete LunarLander PD heuristic (competent landings). Optional only: our data
    characterization suggests the originals were RANDOM-action rollouts (inferred, not confirmed), so
    `--policy random` is the default and the closer match. Returns one of 0=noop, 1=left, 2=main, 3=right."""
    angle_targ = float(np.clip(obs[0] * 0.5 + obs[2] * 1.0, -0.4, 0.4))
    hover_targ = 0.55 * abs(obs[0])
    angle_todo = (angle_targ - obs[4]) * 0.5 - obs[5] * 1.0
    hover_todo = (hover_targ - obs[1]) * 0.5 - obs[3] * 0.5
    if obs[6] or obs[7]:                       # a leg is in contact
        angle_todo = 0.0
        hover_todo = -obs[3] * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def downscale(frame):
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize((IMG_HW[1], IMG_HW[0]), Image.BILINEAR), dtype=np.uint8)


def run_episode(env, rng, policy):
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    imgs, acts, states = [], [], []
    for _ in range(1000):
        imgs.append(downscale(env.render()))
        a = heuristic(obs) if policy == "heuristic" else int(rng.integers(0, 4))
        acts.append(a)
        states.append(np.asarray(obs, dtype=np.float32))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return (np.stack(imgs).astype(np.uint8),
            np.asarray(acts, dtype=np.int32),
            np.stack(states).astype(np.float32))


def write_split(env, rng, n, out_dir, policy):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        imgs, acts, states = run_episode(env, rng, policy)
        np.savez(os.path.join(out_dir, f"{i}.npz"), imgs=imgs, acts=acts, states=states)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {out_dir}: {i + 1}/{n} episodes (last len {len(acts)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=345)
    ap.add_argument("--n_test", type=int, default=55)
    ap.add_argument("--out", default="./data/lunar", help="root; writes lunartrain/ and lunartest/ under it")
    ap.add_argument("--policy", choices=["heuristic", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    try:
        import gymnasium as gym
    except ImportError:
        raise SystemExit('gymnasium is required: pip install "gymnasium[box2d]"')

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    rng = np.random.default_rng(a.seed)
    print(f"generating {a.n_train} train + {a.n_test} test episodes ({a.policy}) -> {a.out}", flush=True)
    write_split(env, rng, a.n_train, os.path.join(a.out, "lunartrain"), a.policy)
    write_split(env, np.random.default_rng(a.seed + 1), a.n_test, os.path.join(a.out, "lunartest"), a.policy)
    env.close()
    print(f"done. data at {a.out} (./data/lunar is the default root; otherwise set PIWM_DATA_ROOT={a.out}).", flush=True)


if __name__ == "__main__":
    main()

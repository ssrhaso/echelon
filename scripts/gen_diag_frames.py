"""
Generate a fixed, policy-neutral held-out frame batch per game for the transfer
diagnostics (D1/D2/D3).

Why random policy + a single fixed batch per game:
  D1/D2/D3 must encode *the same* frames through every condition's encoder so
  that differences in the active code set reflect the codebook/encoder, not the
  policy's exploration. A condition-specific trained policy would confound the
  two. A fixed random-policy batch (seed 42) is policy-neutral and matches the
  sampling protocol already used for Table 1 (game-property measurement).

Frames are stored uint8 (N, 3, 64, 64) exactly as the env emits them; the
diagnostic applies the model preprocessing (float/255 - 0.5) at encode time.

Output: results/diag_frames/<game>.pt   (a uint8 tensor)

Usage:
  python scripts/gen_diag_frames.py --games pong demon_attack ms_pacman up_n_down --n 5000
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnet.envs.atari import AtariEnv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "diag_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the training env config (nnet/models/twister.py:77)
ENV_KW = dict(
    img_size=(64, 64),
    action_repeat=4,
    history_frames=1,
    repeat_action_probability=0.0,
    noop_max=30,
    terminal_on_life_loss=False,
    grayscale_obs=False,
    full_action_space=False,
)


def collect(game: str, n: int, seed: int) -> torch.Tensor:
    env = AtariEnv(game=game, seed=seed, **ENV_KW)
    # numpy RNG for actions, seeded for reproducibility
    g = torch.Generator().manual_seed(seed)
    frames = []
    out = env.reset()
    frames.append(out.state.clone())  # (3,64,64) uint8
    while len(frames) < n:
        a = torch.randint(0, env.num_actions, (), generator=g)
        out = env.step(a)
        frames.append(out.state.clone())
        if out.done.item():
            out = env.reset()
            frames.append(out.state.clone())
    frames = torch.stack(frames[:n], dim=0).to(torch.uint8)  # (N,3,64,64)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["pong", "demon_attack", "ms_pacman", "up_n_down"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for game in args.games:
        dest = OUT_DIR / f"{game}.pt"
        if dest.exists():
            t = torch.load(dest)
            print(f"[skip] {game}: exists {tuple(t.shape)}")
            continue
        print(f"[gen]  {game}: collecting {args.n} frames (seed {args.seed}) ...")
        frames = collect(game, args.n, args.seed)
        torch.save(frames, dest)
        print(f"[ok]   {game}: {tuple(frames.shape)} uint8 -> {dest}")


if __name__ == "__main__":
    main()

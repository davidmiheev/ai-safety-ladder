"""2.2 — Toy Models of Superposition (reproduces Elhage et al., 2022).

Model: n sparse features compressed into m < n hidden dimensions and read back out,
    h = W x,   x' = ReLU(Wᵀ h + b),   loss = E_x Σ_i I_i (x_i − x'_i)²,
with x_i ~ U[0, 1] present independently with probability p = 1 − S (else 0), and
per-feature importance I_i. Many small models (one per sparsity / importance / seed)
are trained in parallel as one batched tensor, so everything runs on a laptop CPU.

Parts (all by default):
  a  n=5, m=2, I_i = 0.9^i: the geometry of W's columns as sparsity rises
  b  n=2, m=1 phase diagram over (importance of feature 2, density)
  c  n=200, m=20 uniform importance: dimensions per feature vs sparsity, and the
     per-feature dimensionality D_i that clusters at "sticky" fractions

Usage:
  uv run python experiments/01_mech_interp/02_toy_superposition.py
  uv run python experiments/01_mech_interp/02_toy_superposition.py --parts a b --steps 5000
  uv run python experiments/01_mech_interp/02_toy_superposition.py --replot   # figures from saved JSON
"""

import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from ladder import get_device, output_dir


def train(n: int, m: int, density: torch.Tensor, importance: torch.Tensor, steps: int,
          batch: int = 1024, lr: float = 1e-3, device: str = "cpu", seed: int = 0):
    """Train k independent toy models at once.

    density: [k] feature probability per model; importance: [k, n].
    Returns W [k, m, n], b [k, n] and the final loss per model [k].
    """
    torch.manual_seed(seed)  # data sampling (global RNG, CPU and CUDA)
    g = torch.Generator(device="cpu").manual_seed(seed)  # weight init
    k = density.shape[0]
    W = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(k, m, n), generator=g).to(device))
    b = torch.nn.Parameter(torch.zeros(k, n, device=device))
    opt = torch.optim.Adam([W, b], lr=lr)
    density, importance = density.to(device), importance.to(device)
    for step in range(steps):
        # Cosine decay to 0 helps the tiny models settle into their final geometry.
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        present = torch.rand(batch, k, n, device=device) < density[None, :, None]
        x = torch.rand(batch, k, n, device=device) * present
        h = torch.einsum("kmn,bkn->bkm", W, x)
        out = torch.relu(torch.einsum("kmn,bkm->bkn", W, h) + b)
        per_model = (importance * (x - out) ** 2).sum(-1).mean(0)  # [k]
        opt.zero_grad()
        per_model.sum().backward()
        opt.step()
    return W.detach().cpu(), b.detach().cpu(), per_model.detach().cpu()


def feature_dimensionality(W: torch.Tensor) -> torch.Tensor:
    """D_i = ||W_i||² / Σ_j (Ŵ_i · W_j)²  for each column i of W [m, n]."""
    norms = W.norm(dim=0)
    unit = W / norms.clamp_min(1e-8)
    overlaps = (unit.T @ W) ** 2  # [n, n]
    return norms**2 / overlaps.sum(-1).clamp_min(1e-12)


# Parts compute results as JSON-serialisable dicts (including what the figures need), and
# separate plot_* functions draw from those dicts. So a run on a remote GPU only has to ship
# back small JSON files, and `--replot` redraws every figure locally.

# --- (a) geometry -----------------------------------------------------------------------------

def part_a(args):
    n, m = 5, 2
    sparsities = [0.0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999]
    k, seeds = len(sparsities), args.seeds
    density = torch.tensor([1 - s for s in sparsities]).repeat(seeds)
    importance = (0.9 ** torch.arange(n)).repeat(k * seeds, 1)
    W, b, loss = train(n, m, density, importance, args.steps, device=args.device)
    # Best of `seeds` restarts per sparsity (single runs sometimes stall in a local minimum).
    loss = loss.view(seeds, k)
    best = loss.argmin(0)
    W = W.view(seeds, k, m, n)[best, torch.arange(k)]
    b = b.view(seeds, k, n)[best, torch.arange(k)]
    loss = loss[best, torch.arange(k)]
    rows = []
    for j, s in enumerate(sparsities):
        norms = W[j].norm(dim=0)
        rows.append({"sparsity": s, "loss": loss[j].item(), "norms": norms.tolist(),
                     "n_represented": int((norms > 0.5).sum()), "bias": b[j].tolist(),
                     "W": W[j].tolist()})
        print(f"  S={s:<6} features represented: {rows[-1]['n_represented']}  "
              f"norms {[round(x, 2) for x in norms.tolist()]}")
    return {"n": n, "m": m, "seeds": seeds, "rows": rows}


def plot_a(res, out):
    rows, n = res["rows"], res["n"]
    fig, axes = plt.subplots(2, len(rows), figsize=(2.3 * len(rows), 4.8),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    colors = plt.cm.viridis(np.linspace(0, 0.9, n))
    for j, r in enumerate(rows):
        W = np.array(r["W"])
        ax = axes[0, j]
        for i in range(n):
            ax.plot([0, W[0, i]], [0, W[1, i]], color=colors[i], lw=2)
            ax.scatter(W[0, i], W[1, i], color=colors[i], s=18, zorder=3)
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xticks=[], yticks=[],
               title=f"S = {r['sparsity']}")
        axes[1, j].imshow(W.T @ W, cmap="RdBu", vmin=-1.2, vmax=1.2)
        axes[1, j].set(xticks=[], yticks=[])
    axes[0, 0].set_ylabel("columns of W")
    axes[1, 0].set_ylabel("WᵀW")
    fig.suptitle("5 features in 2 dimensions (colour: importance, dark = most important)", y=0.99)
    fig.tight_layout()
    fig.savefig(out / "superposition_geometry.png", dpi=150)


# --- (b) phase diagram --------------------------------------------------------------------------

def part_b(args):
    grid = args.grid
    rel_importance = torch.logspace(-1, 1, grid)       # I_2 / I_1
    density = torch.logspace(-2, 0, grid)              # 1 - S
    seeds = args.seeds
    R, D = torch.meshgrid(rel_importance, density, indexing="ij")
    Rf, Df = R.flatten().repeat(seeds), D.flatten().repeat(seeds)
    importance = torch.stack([torch.ones_like(Rf), Rf], dim=-1)
    W, _, loss = train(2, 1, Df, importance, args.steps, device=args.device, seed=1)
    # Keep the best of `seeds` restarts per cell: tiny models do land in local minima.
    loss = loss.view(seeds, grid * grid)
    best = loss.argmin(0)
    Wb = W.view(seeds, grid * grid, 1, 2)[best, torch.arange(grid * grid), 0]  # [cells, 2]

    represented = Wb.abs() > 0.5
    # 0: only feature 1 · 1: only feature 2 · 2: both, in superposition · 3: neither
    phase = torch.full((grid * grid,), 3)
    phase[represented[:, 0] & ~represented[:, 1]] = 0
    phase[~represented[:, 0] & represented[:, 1]] = 1
    phase[represented[:, 0] & represented[:, 1]] = 2
    phase = phase.view(grid, grid)
    antipodal = (Wb[:, 0] * Wb[:, 1] < 0)[represented.all(-1)].float().mean().item()
    counts = {name: int((phase == i).sum()) for i, name in
              enumerate(["only_f1", "only_f2", "superposition", "neither"])}
    print(f"  phase cell counts: {counts}; antipodal among superposed cells: {antipodal:.2f}")
    return {"grid": grid, "seeds": seeds, "counts": counts, "antipodal_fraction": antipodal,
            "rel_importance": rel_importance.tolist(), "density": density.tolist(),
            "phase": phase.tolist()}


def plot_b(res, out):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    cmap = plt.matplotlib.colors.ListedColormap(["#c9c0a8", "#3d5470", "#8a4a2f", "#ffffff"])
    ax.pcolormesh(res["density"], res["rel_importance"], np.array(res["phase"]), cmap=cmap,
                  vmin=-0.5, vmax=3.5, shading="nearest")
    ax.set(xscale="log", yscale="log", xlabel="feature density  1 − S",
           ylabel="relative importance  I₂ / I₁", title="2 features, 1 dimension")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors[:3]]
    ax.legend(handles, ["only feature 1", "only feature 2", "superposition (both)"],
              loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "superposition_phase.png", dpi=150)


# --- (c) dimensionality ------------------------------------------------------------------------

# Feature dimensionalities of the geometric configurations in Elhage et al.:
# dedicated dim, tetrahedron (4 in 3), triangle (3 in 2), digon (2 in 1),
# pentagon (5 in 2), square antiprism (8 in 3).
STICKY = [(1, "1"), (3 / 4, "3/4"), (2 / 3, "2/3"), (1 / 2, "1/2"), (2 / 5, "2/5"), (3 / 8, "3/8")]


def part_c(args):
    n, m = args.n_dim, args.m_dim
    density = torch.logspace(0, -2.5, args.n_sparsities)
    importance = torch.ones(len(density), n)
    t0 = time.time()
    W, _, loss = train(n, m, density, importance, args.steps_c, device=args.device, seed=2)
    print(f"  trained {len(density)} models of {n}->{m} for {args.steps_c} steps in {time.time() - t0:.0f}s")
    dims_per_feature = m / (W**2).sum((1, 2))  # m / ||W||_F^2
    norms = W.norm(dim=1)  # [k, n]
    D = torch.stack([feature_dimensionality(W[j]) for j in range(len(density))])  # [k, n]
    # "Represented" = column norm > 0.5, as in parts (a) and (b).
    represented = norms > 0.5
    D_rep = [[round(v, 4) for v in D[j][represented[j]].tolist()] for j in range(len(density))]

    fracs = torch.tensor([f for f, _ in STICKY])
    Dr = D[represented]
    dist, nearest = (Dr[:, None] - fracs[None]).abs().min(-1)
    close = dist < 0.03
    near = close.float().mean().item() if close.numel() else 0.0
    per_fraction = {label: int((close & (nearest == i)).sum()) for i, (_, label) in enumerate(STICKY)}
    print(f"  represented features within ±0.03 of a sticky fraction: {near:.2f}  {per_fraction}")
    return {"n": n, "m": m, "steps": args.steps_c, "density": density.tolist(),
            "dims_per_feature": dims_per_feature.tolist(),
            "n_represented": represented.sum(-1).tolist(), "D_represented": D_rep,
            "n_represented_features": int(Dr.numel()), "fraction_near_sticky": near,
            "near_sticky_counts": per_fraction, "loss": loss.tolist()}


def plot_c(res, out):
    inv = 1 / np.array(res["density"])
    dpf = np.array(res["dims_per_feature"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    ax1.plot(inv, dpf, "k-", lw=1.5)
    for frac, label in STICKY:
        ax1.axhline(frac, color="grey", lw=0.6, ls=":")
        ax1.text(inv[-1] * 1.05, frac, label, va="center", fontsize=8, color="grey")
    ax1.set(xscale="log", ylim=(0, 1.05), xlabel="1 / (1 − S)", ylabel="m / ‖W‖²_F  (dimensions per feature)",
            title=f"{res['n']} features in {res['m']} dimensions")
    for j, Ds in enumerate(res["D_represented"]):
        ax2.scatter(np.full(len(Ds), inv[j]), Ds, s=3, alpha=0.35, color="#3d5a4a")
    ax2.plot(inv, dpf, "k-", lw=1)
    for frac, _ in STICKY:
        ax2.axhline(frac, color="grey", lw=0.6, ls=":")
    ax2.set(xscale="log", ylim=(0, 1.1), xlabel="1 / (1 − S)", ylabel="feature dimensionality  Dᵢ",
            title="per-feature dimensionality (features with ‖Wᵢ‖ > 0.5)")
    fig.tight_layout()
    fig.savefig(out / "superposition_dimensionality.png", dpi=150)


PARTS = {"a": (part_a, plot_a), "b": (part_b, plot_b), "c": (part_c, plot_c)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["a", "b", "c"], choices=list(PARTS))
    parser.add_argument("--replot", action="store_true", help="redraw figures from saved JSON, no training")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--steps-c", type=int, default=50_000)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n-dim", type=int, default=200)
    parser.add_argument("--m-dim", type=int, default=20)
    parser.add_argument("--n-sparsities", type=int, default=40)
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()

    out = output_dir("01_mech_interp")
    for part in args.parts:
        compute, plot = PARTS[part]
        path = out / f"toy_superposition_{part}.json"
        if args.replot:
            res = json.loads(path.read_text())
        else:
            print(f"part {part}:")
            t0 = time.time()
            res = compute(args)
            print(f"  ({time.time() - t0:.0f}s)")
            path.write_text(json.dumps(res))
        plot(res, out)
        print(f"  {path.name} + figure written")


if __name__ == "__main__":
    main()

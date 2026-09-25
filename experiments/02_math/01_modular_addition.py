"""3.1 — Modular addition: grokking and the Fourier (circular) embedding.

Reproduces Nanda et al. (2023), "Progress measures for grokking via mechanistic
interpretability", in miniature. A 1-layer transformer learns (a + b) mod p from 30% of all
pairs. It first memorises (train loss → 0, test loss stays high), then *groks* (test loss
collapses). The generalising algorithm is trigonometric: embed a as (cos ωa, sin ωa) for a few
key frequencies ω = 2πk/p, use the MLP to form cos ω(a+b), sin ω(a+b), and read out
logit(c) ∝ Σ_ω cos ω(a + b − c).

What this script measures:
  * train/test loss over training (the grokking curve);
  * the Fourier spectrum of the token embedding: norm of W_E along cos/sin of each frequency;
  * the circle: W_E projected onto the (cos, sin) pair of a key frequency, points ordered by k·a mod p;
  * a progress measure over training: the share of the embedding's norm in its top frequencies,
    and the test loss of the model *restricted* to those frequencies (Nanda's "restricted loss").

~5–10 min on CPU.
  uv run python experiments/02_math/01_modular_addition.py
  uv run python experiments/02_math/01_modular_addition.py --replot
"""

import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ladder import get_device, output_dir


def fourier_basis(p: int) -> tuple[torch.Tensor, list[str]]:
    """Orthonormal real Fourier basis on Z_p: constant, then cos/sin for k = 1 .. (p-1)/2."""
    x = torch.arange(p, dtype=torch.float64)
    rows, names = [torch.ones(p, dtype=torch.float64)], ["const"]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * x / p), torch.sin(2 * math.pi * k * x / p)]
        names += [f"cos {k}", f"sin {k}"]
    B = torch.stack(rows)
    return (B / B.norm(dim=1, keepdim=True)).float(), names


def make_model(p: int, device: str, seed: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=128, d_head=32, n_heads=4, d_mlp=512, d_vocab=p + 1, d_vocab_out=p,
        n_ctx=3, act_fn="relu", normalization_type=None, device=device, seed=seed,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():  # Nanda et al.: no biases (b_Q, b_in, b_U, ...)
        if name.split(".")[-1].startswith("b_"):
            param.requires_grad_(False)
    return model


def embedding_spectrum(W_E: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Squared norm of the embedding along each frequency k (cos and sin combined). [n_freq]"""
    coef = B @ W_E  # [p, d]: row i = component of the embedding along basis vector i
    sq = coef.pow(2).sum(-1)
    return torch.stack([sq[1 + 2 * (k - 1)] + sq[2 + 2 * (k - 1)] for k in range(1, (len(sq) - 1) // 2 + 1)])


def restricted_loss(model, tokens, labels, B, key_freqs, p) -> float:
    """Test loss when the logits are projected onto the key frequencies of (a + b − c)."""
    logits = model(tokens)[:, -1].double()  # [n, p]
    a, b = tokens[:, 0], tokens[:, 1]
    c = torch.arange(p, device=logits.device)
    kept = torch.zeros_like(logits)
    mean = logits.mean(-1, keepdim=True)
    for k in key_freqs:
        w = 2 * math.pi * k / p
        arg = w * (a + b)[:, None] - w * c[None]
        cos_c, sin_c = torch.cos(arg), torch.sin(arg)
        # Least-squares coefficients of logits on cos/sin of ω(a+b−c), per example.
        kept += ((logits - mean) * cos_c).sum(-1, keepdim=True) / cos_c.pow(2).sum(-1, keepdim=True) * cos_c
        kept += ((logits - mean) * sin_c).sum(-1, keepdim=True) / sin_c.pow(2).sum(-1, keepdim=True) * sin_c
    kept += mean
    return torch.nn.functional.cross_entropy(kept, labels).item()


def logit_spectrum(logits: torch.Tensor, p: int) -> torch.Tensor:
    """Mean power of the logit vector over answers c at each frequency k = 1..(p-1)/2.

    logits: [n, p]. Power at k combines the cos and sin components (rfft bin k)."""
    spec = torch.fft.rfft(logits.double() - logits.double().mean(-1, keepdim=True), dim=-1).abs().pow(2)
    return spec[:, 1:p // 2 + 1].mean(0)


def keep_freqs_loss(logits: torch.Tensor, labels: torch.Tensor, freqs: list[int], p: int, keep: bool) -> float:
    """Loss after keeping (keep=True) or removing (keep=False) the given frequencies of each
    example's logit vector over c (plus the mean)."""
    F = torch.fft.rfft(logits.double(), dim=-1)
    mask = torch.zeros(F.shape[-1], dtype=torch.bool, device=F.device)
    mask[freqs] = True
    mask[0] = True  # the mean never matters for the softmax
    F = F * (mask if keep else ~mask | (torch.arange(F.shape[-1], device=F.device) == 0))
    return torch.nn.functional.cross_entropy(torch.fft.irfft(F, n=p, dim=-1), labels).item()


def train(args) -> dict:
    torch.manual_seed(args.seed)
    p, device = args.p, args.device
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    tokens = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], dim=1).to(device)  # "a b ="
    labels = ((a + b) % p).flatten().to(device)
    perm = torch.randperm(p * p)
    n_train = int(args.frac * p * p)
    tr, te = perm[:n_train], perm[n_train:]

    model = make_model(p, device, args.seed)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    B, _ = fourier_basis(p)
    B = B.to(device)

    hist = {"step": [], "train_loss": [], "test_loss": [], "top_freq_share": [], "restricted_loss": [],
            "weight_norm": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        loss = torch.nn.functional.cross_entropy(model(tokens[tr])[:, -1], labels[tr])
        if step % args.log_every == 0:
            with torch.no_grad():
                test_loss = torch.nn.functional.cross_entropy(model(tokens[te])[:, -1], labels[te]).item()
                spec = embedding_spectrum(model.W_E[:p].detach(), B)
                top = spec.topk(args.n_key).indices + 1
                hist["step"].append(step)
                hist["train_loss"].append(loss.item())
                hist["test_loss"].append(test_loss)
                hist["top_freq_share"].append((spec.topk(args.n_key).values.sum() / spec.sum()).item())
                hist["restricted_loss"].append(restricted_loss(model, tokens[te], labels[te], B, top.tolist(), p))
                hist["weight_norm"].append(sum(q.pow(2).sum() for q in params).sqrt().item())
            if step % (args.log_every * 20) == 0:
                print(f"  step {step:>6}  train {loss.item():.4f}  test {test_loss:.4f}  "
                      f"top-{args.n_key} freq share {hist['top_freq_share'][-1]:.2f}  ({time.time() - t0:.0f}s)")
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        W_E = model.W_E[:p].detach()
        spec = embedding_spectrum(W_E, B)
        key = (spec.topk(args.n_key).indices + 1).tolist()
        acc = (model(tokens[te])[:, -1].argmax(-1) == labels[te]).float().mean().item()
        # Circle for the strongest frequency: projections on its cos / sin basis vectors.
        k0 = key[0]
        circle = torch.stack([B[1 + 2 * (k0 - 1)] @ W_E, B[2 + 2 * (k0 - 1)] @ W_E])  # [2, d]
        # Coordinates of each token a: project W_E[a] onto the 2D subspace spanned by those directions.
        basis, _ = torch.linalg.qr(circle.T)  # [d, 2]
        xy = (W_E @ basis).cpu()
        # Neurons: fraction of each MLP neuron's activation variance explained by its best key frequency pair.
        all_logits, cache = model.run_with_cache(tokens)
        acts = cache["blocks.0.mlp.hook_post"][:, -1].reshape(p, p, -1)  # [a, b, neuron]
        # Output side: which frequencies do the logits (over answers c) actually use?
        test_logits = all_logits[te, -1]
        lspec = logit_spectrum(test_logits, p)
        logit_key = (lspec.topk(args.n_key).indices + 1).tolist()
        output_side = {
            "logit_spectrum": (lspec / lspec.sum()).cpu().tolist(),
            "logit_top_freqs": logit_key,
            "logit_share_in_embedding_keys": (lspec[[k - 1 for k in key]].sum() / lspec.sum()).item(),
            "logit_share_in_logit_top": (lspec.topk(args.n_key).values.sum() / lspec.sum()).item(),
            "restricted_loss_embedding_keys": keep_freqs_loss(test_logits, labels[te], key, p, keep=True),
            "restricted_loss_logit_keys": keep_freqs_loss(test_logits, labels[te], logit_key, p, keep=True),
            "excluded_loss_logit_keys": keep_freqs_loss(test_logits, labels[te], logit_key, p, keep=False),
            "test_loss": torch.nn.functional.cross_entropy(test_logits, labels[te]).item(),
        }
        torch.save(model.state_dict(), output_dir("02_math") / "modular_addition_model.pt")
    return {
        "p": p, "frac": args.frac, "steps": args.steps, "seed": args.seed, "history": hist,
        "final_test_accuracy": acc, "spectrum": spec.cpu().tolist(), "key_freqs": key,
        "circle_freq": k0, "circle_xy": xy.tolist(),
        "neuron_freq_r2": neuron_frequency_r2(acts.cpu(), p, key),
        "output_side": output_side,
    }


def neuron_frequency_r2(acts: torch.Tensor, p: int, key: list[int]) -> dict:
    """For each neuron, the best R² of its (a, b) activation map on {cos, sin} of ω(a+b) plus the
    single-input terms, over key frequencies. High R² = the neuron computes a function of a+b."""
    a = torch.arange(p, dtype=torch.float64)[:, None].expand(p, p)
    b = torch.arange(p, dtype=torch.float64)[None, :].expand(p, p)
    best = torch.zeros(acts.shape[-1], dtype=torch.float64)
    Y = acts.reshape(p * p, -1).double()
    Y = Y - Y.mean(0)
    for k in key:
        w = 2 * math.pi * k / p
        X = torch.stack([torch.cos(w * (a + b)), torch.sin(w * (a + b)), torch.cos(w * a), torch.sin(w * a),
                         torch.cos(w * b), torch.sin(w * b)], -1).reshape(p * p, -1)
        X = X - X.mean(0)
        coef = torch.linalg.lstsq(X, Y).solution
        r2 = 1 - (Y - X @ coef).pow(2).sum(0) / Y.pow(2).sum(0).clamp_min(1e-12)
        best = torch.maximum(best, r2)
    alive = Y.pow(2).sum(0) > 1e-8
    b_alive = best[alive]
    return {"n_neurons": int(alive.sum()), "median_r2": b_alive.median().item(),
            "frac_above_0.9": (b_alive > 0.9).double().mean().item()}


def plot(res: dict, out) -> None:
    h = res["history"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(h["step"], h["train_loss"], label="train")
    ax.plot(h["step"], h["test_loss"], label="test")
    ax.plot(h["step"], h["restricted_loss"], label=f"test, restricted to top-{len(res['key_freqs'])} freqs", ls="--")
    ax.set(yscale="log", xlabel="step", ylabel="cross-entropy", title="grokking")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(h["step"], h["top_freq_share"], color="#3d5a4a", label="embedding norm in top freqs")
    ax.set(xlabel="step", ylabel="share", ylim=(0, 1.02), title="progress measures")
    ax2 = ax.twinx()
    ax2.plot(h["step"], h["weight_norm"], color="#8a4a2f", label="total weight norm")
    ax2.set_ylabel("weight norm")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    ax = axes[1, 0]
    spec = np.array(res["spectrum"])
    ks = np.arange(1, len(spec) + 1)
    colors = ["#8a4a2f" if k in res["key_freqs"] else "#c9c0a8" for k in ks]
    ax.bar(ks, spec / spec.sum(), color=colors)
    ax.set(xlabel="frequency k", ylabel="share of ‖W_E‖²", title="embedding Fourier spectrum (key frequencies highlighted)")
    ax = axes[1, 1]
    xy = np.array(res["circle_xy"])
    p, k0 = res["p"], res["circle_freq"]
    order = (np.arange(p) * k0) % p
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=order, cmap="twilight", s=22)
    for a in range(0, p, max(1, p // 12)):
        ax.annotate(str(a), xy[a], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(aspect="equal", title=f"W_E on the frequency-{k0} plane (colour: {k0}·a mod {p})", xticks=[], yticks=[])
    plt.colorbar(sc, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "modular_addition.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--frac", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-key", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()

    out = output_dir("02_math")
    path = out / "modular_addition.json"
    if args.replot:
        res = json.loads(path.read_text())
    else:
        res = train(args)
        path.write_text(json.dumps(res))
        h = res["history"]
        grok = next((s for s, t in zip(h["step"], h["test_loss"]) if t < 0.1), None)
        print(f"test accuracy {res['final_test_accuracy']:.4f}; key frequencies {res['key_freqs']}; "
              f"test loss < 0.1 from step {grok}; neurons: {res['neuron_freq_r2']}")
        print("output side:", {k: v for k, v in res["output_side"].items() if k != "logit_spectrum"})
    plot(res, out)
    print(f"-> {out / 'modular_addition.png'}, {path}")


if __name__ == "__main__":
    main()

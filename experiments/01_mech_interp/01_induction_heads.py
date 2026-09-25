"""2.1 — Find the induction heads in GPT-2 small (reproduces Olsson et al., 2022).

Setup: a sequence of random tokens repeated twice, [BOS, A, A]. On the second copy
the model can predict every token by "look back to where the current token last
appeared, copy what came next" — the induction mechanism. Nothing else helps,
since the tokens are random.

Three lines of evidence, strongest last:
  1. Behaviour: per-token loss collapses on the second copy (in-context learning).
  2. Attention: induction heads attend from position i to i - L + 1 (the token
     *after* the previous occurrence); previous-token heads attend to i - 1.
  3. Causality (activation patching): run a *corrupted* input whose first half is
     different random tokens (so the second half is no longer a repeat), patch in
     one head's clean output, and measure how much second-half log-prob that one
     head restores. Patching the head's output (hook_z) at every position, so a
     previous-token head acting on the first half is credited too.
  4. Where vs what: induction heads need two things from the first copy — *where*
     to look (K-composition with previous-token heads writing "my previous token
     was X" into each position) and *what* to copy (the token identity there, read
     by the OV circuit). Patching only previous-token heads restores the attention
     but copies corrupted tokens; patching token-identity components restores the
     prediction.

Numbers are also written to outputs/01_mech_interp/induction_heads.json for the write-up.

Usage:
  uv run python experiments/01_mech_interp/01_induction_heads.py
  uv run python experiments/01_mech_interp/01_induction_heads.py --seq-len 50 --batch 16
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from ladder import get_device, output_dir

# Heads reported for GPT-2 small in the literature, for comparison with what we find.
KNOWN_INDUCTION = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}


def repeated_tokens(model, seq_len: int, batch: int, generator) -> torch.Tensor:
    rand = torch.randint(1000, 20000, (batch, seq_len), generator=generator)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id)
    return torch.cat([bos, rand, rand], dim=1).to(model.cfg.device)


def second_half_logprob(logits: torch.Tensor, tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Mean log-prob of the correct next token over predictions made inside the second copy."""
    logprobs = logits.log_softmax(-1)[:, :-1]
    correct = logprobs.gather(-1, tokens[:, 1:, None])[..., 0]  # [batch, pos]
    return correct[:, seq_len + 1:].mean()  # predictions for tokens 2..L of the second copy


def attention_scores(cache, n_layers: int, seq_len: int):
    """Per-head average attention to (i - L + 1) [induction] and (i - 1) [previous token]."""
    induction, prev = [], []
    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # [batch, head, q, k]
        induction.append(pattern.diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)[..., 1:].mean((0, -1)))
        prev.append(pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean((0, -1)))
    return torch.stack(induction), torch.stack(prev)  # [layer, head]


def patch_heads(model, clean_cache, corrupt_tokens, seq_len: int) -> torch.Tensor:
    """Log-prob recovered by patching each head's clean output into the corrupted run."""
    cfg = model.cfg
    result = torch.zeros(cfg.n_layers, cfg.n_heads)
    for layer in range(cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        clean_z = clean_cache[hook_name]
        for head in range(cfg.n_heads):
            def hook(z, hook, head=head, clean_z=clean_z):
                z[:, :, head] = clean_z[:, :, head]
                return z

            logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, hook)])
            result[layer, head] = second_half_logprob(logits, corrupt_tokens, seq_len)
    return result


def induction_attention(cache, heads, seq_len: int) -> float:
    """Mean attention of `heads` to the induction target (i - L + 1) on the second copy."""
    return sum(
        cache["pattern", layer][:, head].diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)[..., 1:].mean().item()
        for layer, head in heads
    ) / len(heads)


def patched_run(model, corrupt_tokens, hooks, seq_len: int, heads):
    """(second-half log-prob, induction attention of `heads`) for a corrupted run with `hooks`."""
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(corrupt_tokens, names_filter=lambda n: n.endswith("pattern"))
    return second_half_logprob(logits, corrupt_tokens, seq_len).item(), induction_attention(cache, heads, seq_len)


def head_hooks(clean_cache, heads):
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    hooks = []
    for layer, hs in by_layer.items():
        name = f"blocks.{layer}.attn.hook_z"

        def hook(z, hook, hs=hs, name=name):
            z[:, :, hs] = clean_cache[name][:, :, hs]
            return z

        hooks.append((name, hook))
    return hooks


def first_half_hooks(clean_cache, names, seq_len: int):
    """Patch whole activations, but only at the first-copy positions 1..L."""
    def make(name):
        def hook(x, hook):
            x[:, 1:seq_len + 1] = clean_cache[name][:, 1:seq_len + 1]
            return x
        return hook
    return [(name, make(name)) for name in names]


def heatmap(ax, data, title, cmap="Blues", center=False):
    vmax = data.abs().max()
    kw = {"cmap": "RdBu", "vmin": -vmax, "vmax": vmax} if center else {"cmap": cmap}
    im = ax.imshow(data.numpy(), aspect="auto", **kw)
    ax.set(title=title, xlabel="head", ylabel="layer",
           xticks=range(data.shape[1]), yticks=range(data.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046)


def top(scores, k=6):
    flat = scores.flatten().topk(k)
    n_heads = scores.shape[1]
    return [(i.item() // n_heads, i.item() % n_heads, v.item()) for v, i in zip(flat.values, flat.indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    L = args.seq_len

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device=get_device())
    gen = torch.Generator().manual_seed(args.seed)

    clean = repeated_tokens(model, L, args.batch, gen)
    # Corrupted: first copy replaced by fresh random tokens -> second half is no longer a repeat.
    corrupt = clean.clone()
    corrupt[:, 1:L + 1] = torch.randint(1000, 20000, (args.batch, L), generator=gen).to(clean.device)

    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits = model(corrupt)

    # --- 1. Behaviour --------------------------------------------------------------
    loss = -clean_logits.log_softmax(-1)[:, :-1].gather(-1, clean[:, 1:, None])[..., 0].mean(0).cpu()
    print(f"mean loss, first copy: {loss[:L].mean():.2f} nats | second copy: {loss[L + 1:].mean():.2f} nats")

    # --- 2. Attention patterns -------------------------------------------------------
    induction, prev = attention_scores(clean_cache, model.cfg.n_layers, L)
    induction, prev = induction.cpu(), prev.cpu()
    print("\ntop induction-score heads (attn to token after previous occurrence):")
    for layer, head, v in top(induction):
        known = "  <- known" if (layer, head) in KNOWN_INDUCTION else ""
        print(f"  {layer}.{head:<3} {v:.2f}{known}")
    print("top previous-token heads:")
    for layer, head, v in top(prev, 4):
        print(f"  {layer}.{head:<3} {v:.2f}")

    # --- 3. Activation patching -------------------------------------------------------
    clean_lp = second_half_logprob(clean_logits, clean, L).item()
    corrupt_lp = second_half_logprob(corrupt_logits, corrupt, L).item()
    print(f"\nsecond-half log-prob: clean {clean_lp:.2f}, corrupted {corrupt_lp:.2f}")
    patched = patch_heads(model, clean_cache, corrupt, L)
    recovered = (patched - corrupt_lp) / (clean_lp - corrupt_lp)
    print("heads whose clean output alone recovers the most (fraction of clean-corrupt gap):")
    for layer, head, v in top(recovered):
        print(f"  {layer}.{head:<3} {v:+.3f}  (induction score {induction[layer, head]:.2f}, "
              f"prev-token score {prev[layer, head]:.2f})")

    # Single heads are weak because the circuit is redundant; patch it as a unit.
    top_prev = [(layer, head) for layer, head, _ in top(prev, 1)]
    top_ind = [(layer, head) for layer, head, _ in top(induction, 5)]
    print("\npatching head *sets* (fraction of gap recovered):")
    head_sets = []
    for label, heads in [("previous-token head", top_prev),
                         ("top-5 induction heads", top_ind),
                         ("prev-token + top-5 induction", top_prev + top_ind)]:
        lp, _ = patched_run(model, corrupt, head_hooks(clean_cache, heads), L, top_ind)
        recovered_set = (lp - corrupt_lp) / (clean_lp - corrupt_lp)
        names = " ".join(f"{la}.{h}" for la, h in heads)
        head_sets.append({"set": label, "heads": names, "recovered": recovered_set})
        print(f"  {label:<30} {recovered_set:+.3f}   [{names}]")

    # --- 4. Where vs what -----------------------------------------------------------------
    gap = clean_lp - corrupt_lp

    def frac(lp):
        return (lp - corrupt_lp) / gap

    first_layer = min(layer for layer, _ in top_ind)
    early_prev = sorted(((prev[la, h].item(), la, h) for la in range(first_layer)
                         for h in range(model.cfg.n_heads)), reverse=True)
    sweep = []
    print(f"\nwhere vs what — patch the k strongest previous-token heads in layers < {first_layer}:")
    for k in (1, 3, 5, 10, 20, len(early_prev)):
        heads = [(la, h) for _, la, h in early_prev[:k]]
        lp, attn = patched_run(model, corrupt, head_hooks(clean_cache, heads), L, top_ind)
        sweep.append({"k": k, "induction_attention": attn, "recovered": frac(lp)})
        print(f"  k={k:<3} induction attention {attn:.2f}   log-prob recovered {frac(lp):+.3f}")

    _, base_attn = patched_run(model, corrupt, [], L, top_ind)
    early = range(first_layer)
    components = {
        "attention outputs (all heads)": [f"blocks.{la}.hook_attn_out" for la in early],
        "MLP 0 output": ["blocks.0.hook_mlp_out"],
        "MLP outputs": [f"blocks.{la}.hook_mlp_out" for la in early],
        "attention + MLP 0": [f"blocks.{la}.hook_attn_out" for la in early] + ["blocks.0.hook_mlp_out"],
        "attention + MLP outputs": [f"blocks.{la}.hook_{k}_out" for la in early for k in ("attn", "mlp")],
    }
    decomposition = []
    print(f"component patching, first-copy positions only, layers < {first_layer}:")
    for label, names in components.items():
        lp, attn = patched_run(model, corrupt, first_half_hooks(clean_cache, names, L), L, top_ind)
        decomposition.append({"component": label, "induction_attention": attn, "recovered": frac(lp)})
        print(f"  {label:<32} induction attention {attn:.2f}   log-prob recovered {frac(lp):+.3f}")

    # --- Figures ------------------------------------------------------------------------
    out = output_dir("01_mech_interp")
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    axes[0].plot(loss.numpy())
    axes[0].axvline(L, color="grey", ls="--")
    axes[0].set(title="per-token loss on [BOS, A, A]", xlabel="position", ylabel="loss (nats)")
    heatmap(axes[1], induction, "induction score")
    heatmap(axes[2], prev, "previous-token score")
    heatmap(axes[3], recovered, "patching: fraction recovered", center=True)
    fig.tight_layout()
    path = out / "induction_heads.png"
    fig.savefig(path, dpi=150)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ks = [r["k"] for r in sweep]
    ax1.plot(ks, [r["induction_attention"] for r in sweep], "o-", label="induction attention")
    ax1.plot(ks, [r["recovered"] for r in sweep], "s-", label="log-prob recovered")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set(xscale="log", xlabel=f"# previous-token heads patched (layers < {first_layer})",
            title="restoring *where* to look is not enough")
    ax1.legend()
    labels = [r["component"] for r in decomposition]
    y = range(len(labels))
    ax2.barh([i + 0.2 for i in y], [r["induction_attention"] for r in decomposition], 0.4, label="induction attention")
    ax2.barh([i - 0.2 for i in y], [r["recovered"] for r in decomposition], 0.4, label="log-prob recovered")
    ax2.set(yticks=list(y), yticklabels=labels, xlim=(-0.1, 1.05),
            title=f"patching first-copy components (layers < {first_layer})")
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    path2 = out / "induction_where_vs_what.png"
    fig2.savefig(path2, dpi=150)

    results = {
        "seq_len": L, "batch": args.batch, "seed": args.seed,
        "loss_first_copy": loss[:L].mean().item(), "loss_second_copy": loss[L + 1:].mean().item(),
        "top_induction": [{"head": f"{la}.{h}", "score": v, "known": (la, h) in KNOWN_INDUCTION}
                          for la, h, v in top(induction)],
        "top_prev_token": [{"head": f"{la}.{h}", "score": v} for la, h, v in top(prev, 4)],
        "clean_logprob": clean_lp, "corrupt_logprob": corrupt_lp,
        "top_patching": [{"head": f"{la}.{h}", "recovered": v} for la, h, v in top(recovered)],
        "most_negative_patching": [{"head": f"{la}.{h}", "recovered": -v} for la, h, v in top(-recovered, 3)],
        "head_sets": head_sets,
        "corrupted_induction_attention": base_attn,
        "prev_token_sweep": sweep,
        "component_decomposition": decomposition,
    }
    (out / "induction_heads.json").write_text(json.dumps(results, indent=2))
    print(f"\nfigures -> {path}, {path2}\nnumbers -> {out / 'induction_heads.json'}")


if __name__ == "__main__":
    main()

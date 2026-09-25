"""Setup 1 — TransformerLens + CircuitsVis: first forward pass with activation caching.

Goals:
  * load a small model as a HookedTransformer
  * run it once with `run_with_cache` and walk the cache (every layer's activations)
  * check you understand the residual stream: rebuilding the logits from the
    cached final residual gives the same answer as the model
  * render attention patterns with CircuitsVis to an HTML file

Usage:
  uv run python experiments/00_setup/01_transformerlens_first_pass.py
  uv run python experiments/00_setup/01_transformerlens_first_pass.py --model pythia-70m
"""

import argparse
import json

import circuitsvis as cv
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from ladder import get_device, output_dir

PROMPT = "When Mary and John went to the store, John gave a drink to"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small", help="e.g. gpt2-small, pythia-70m")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=0, help="layer to visualise attention for")
    parser.add_argument("--head", default="9.9", help="head L.H to plot statically (9.9: an IOI name mover)")
    parser.add_argument("--also", default=" John", help="extra token to track in the logit-lens plot")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = get_device()
    model = HookedTransformer.from_pretrained(args.model, device=device)
    cfg = model.cfg
    print(f"{args.model}: {cfg.n_layers} layers, {cfg.n_heads} heads, d_model={cfg.d_model}, "
          f"d_vocab={cfg.d_vocab} on {device}")

    tokens = model.to_tokens(args.prompt)
    str_tokens = model.to_str_tokens(args.prompt)
    logits, cache = model.run_with_cache(tokens)

    # --- 1. What's in the cache? -------------------------------------------------
    print(f"\ncache holds {len(cache)} activations; layer-0 hook points:")
    for name, act in cache.items():
        if name.startswith("blocks.0."):
            print(f"  {name:<32} {tuple(act.shape)}")

    # --- 2. Next-token prediction -------------------------------------------------
    probs = logits[0, -1].softmax(-1)
    top = probs.topk(5)
    print(f"\nprompt: {args.prompt!r}")
    for p, idx in zip(top.values, top.indices):
        print(f"  {model.to_string(idx)!r:>12}  {p.item():.3f}")

    # --- 3. Residual stream sanity check -------------------------------------------
    # Final residual -> ln_final -> unembed should reproduce the model's logits.
    resid_final = cache["resid_post", cfg.n_layers - 1]
    rebuilt = model.unembed(model.ln_final(resid_final))
    max_err = (rebuilt - logits).abs().max().item()
    print(f"\nlogits rebuilt from cached residual stream: max |diff| = {max_err:.2e}")
    assert max_err < 1e-3, "residual-stream reconstruction should match the forward pass"

    # --- 4. Logit lens: when does the answer appear? ------------------------------
    answer = top.indices[0]
    accum, labels = cache.accumulated_resid(apply_ln=True, pos_slice=-1, return_labels=True)
    lens_logits = model.unembed(accum)[:, 0]  # [component, d_vocab]
    ranks = (lens_logits > lens_logits[:, answer, None]).sum(-1)
    print(f"\nlogit lens — rank of {model.to_string(answer)!r} after each layer:")
    print("  " + "  ".join(f"{lbl}:{r.item()}" for lbl, r in zip(labels, ranks)))
    lens_probs = lens_logits.softmax(-1)
    rivals = top.indices[:2].tolist()  # ' Mary', ' them' on the default prompt
    extra = model.to_single_token(args.also) if args.also else None
    if extra is not None and extra not in rivals:
        rivals.append(extra)  # the subject ' John' — the name the model must *not* predict

    # --- 5. CircuitsVis attention patterns -----------------------------------------
    pattern = cache["pattern", args.layer][0]  # [head, query, key]
    html = cv.attention.attention_patterns(tokens=str_tokens, attention=pattern)
    out = output_dir("00_setup") / f"attention_{args.model}_L{args.layer}.html"
    out.write_text(str(html))
    print(f"\nattention patterns for layer {args.layer} -> {out}")

    # --- 6. Static figure + numbers for the write-up -------------------------------------
    hl, hh = map(int, args.head.split("."))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for idx in rivals:
        ax1.plot(lens_probs[:, idx].cpu().numpy(), "o-", label=repr(model.to_string(idx)))
    ax1.set(xticks=range(len(labels)), xlabel="residual stream after", ylabel="probability",
            title="logit lens at the final position")
    ax1.set_xticklabels(labels, rotation=60, fontsize=8)
    ax1.legend()
    head_pattern = cache["pattern", hl][0, hh].cpu()
    ax2.imshow(head_pattern.numpy(), cmap="Blues")
    ax2.set(xticks=range(len(str_tokens)), yticks=range(len(str_tokens)), title=f"attention of head {hl}.{hh}")
    ax2.set_xticklabels(str_tokens, rotation=90, fontsize=7)
    ax2.set_yticklabels(str_tokens, fontsize=7)
    ax2.set(xlabel="key", ylabel="query")
    fig.tight_layout()
    fig_path = output_dir("00_setup") / f"logit_lens_{args.model}.png"
    fig.savefig(fig_path, dpi=150)

    results = {
        "model": args.model, "prompt": args.prompt,
        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "d_model": cfg.d_model, "d_vocab": cfg.d_vocab,
        "n_cached_activations": len(cache),
        "top_predictions": [{"token": model.to_string(i), "prob": p.item()} for p, i in zip(top.values, top.indices)],
        "resid_reconstruction_max_err": max_err,
        "logit_lens_rank": {lbl: r.item() for lbl, r in zip(labels, ranks)},
        "logit_lens_prob": {model.to_string(i): lens_probs[:, i].tolist() for i in rivals},
        "logit_lens_labels": labels,
        "static_head": args.head,
        "static_head_final_query_top_key": str_tokens[head_pattern[-1].argmax().item()],
        "static_head_final_query_top_weight": head_pattern[-1].max().item(),
    }
    json_path = output_dir("00_setup") / f"first_pass_{args.model}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"figure -> {fig_path}\nnumbers -> {json_path}")


if __name__ == "__main__":
    main()

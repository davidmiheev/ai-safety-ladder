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

import circuitsvis as cv
import torch
from transformer_lens import HookedTransformer

from ladder import get_device, output_dir

PROMPT = "When Mary and John went to the store, John gave a drink to"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small", help="e.g. gpt2-small, pythia-70m")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layer", type=int, default=0, help="layer to visualise attention for")
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

    # --- 5. CircuitsVis attention patterns -----------------------------------------
    pattern = cache["pattern", args.layer][0]  # [head, query, key]
    html = cv.attention.attention_patterns(tokens=str_tokens, attention=pattern)
    out = output_dir("00_setup") / f"attention_{args.model}_L{args.layer}.html"
    out.write_text(str(html))
    print(f"\nattention patterns for layer {args.layer} -> {out}")


if __name__ == "__main__":
    main()

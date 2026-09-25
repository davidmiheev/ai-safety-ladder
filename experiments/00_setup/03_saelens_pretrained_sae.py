"""Setup 3 — SAELens + Neuronpedia: load a pretrained SAE on GPT-2 small and look at features.

Before training your own SAE (section 2), see what a good one already found.

Goals:
  * load Joseph Bloom's residual-stream SAE for GPT-2 small (layer 8 by default)
  * check reconstruction quality: L0, explained variance, and the CE-loss hit when
    the SAE reconstruction is spliced back into the forward pass
  * list the top-firing features on a few prompts, with Neuronpedia links
  * then pick 5 and write them up by hand in notes/sae_features_pretrained.md

Usage:
  uv run python experiments/00_setup/03_saelens_pretrained_sae.py
  uv run python experiments/00_setup/03_saelens_pretrained_sae.py --layer 6 --prompt "your text"
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch
from sae_lens import SAE, HookedSAETransformer

from ladder import get_device, output_dir

RELEASE = "gpt2-small-res-jb"
NEURONPEDIA = "https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{idx}"
PROMPTS = [
    "The Golden Gate Bridge is a suspension bridge in San Francisco.",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "She was furious, slammed the door, and refused to speak to anyone.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8, help="0-11; hook is blocks.L.hook_resid_pre")
    parser.add_argument("--prompt", action="append", help="repeatable; defaults to built-in prompts")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = get_device()
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    sae = SAE.from_pretrained(RELEASE, f"blocks.{args.layer}.hook_resid_pre", device=device)
    hook = sae.cfg.metadata.hook_name
    print(f"SAE {RELEASE} @ {hook}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    results = {"release": RELEASE, "hook": hook, "d_in": sae.cfg.d_in, "d_sae": sae.cfg.d_sae, "prompts": []}
    heat = None
    for prompt in args.prompt or PROMPTS:
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        _, cache = model.run_with_cache(tokens, names_filter=hook)
        resid = cache[hook][0, 1:]  # drop BOS: its norm is huge and it isn't what we care about
        feats = sae.encode(resid)
        recon = sae.decode(feats)

        l0 = (feats > 0).sum(-1).float().mean().item()
        fvu = ((resid - recon).pow(2).sum() / (resid - resid.mean(0)).pow(2).sum()).item()
        clean_loss = model(tokens, return_type="loss").item()
        spliced_loss = model.run_with_saes(tokens, saes=[sae], return_type="loss").item()

        print(f"\n{prompt!r}")
        print(f"  L0={l0:.1f}  explained var={1 - fvu:.3f}  "
              f"CE loss {clean_loss:.3f} -> {spliced_loss:.3f} with SAE spliced in")

        # Strongest features anywhere in the prompt, and where they peak.
        peak, pos = feats.max(0)
        top_feats = []
        for act, idx in zip(*peak.topk(args.top_k)):
            tok = str_tokens[1 + pos[idx].item()]
            url = NEURONPEDIA.format(layer=args.layer, idx=idx.item())
            print(f"  feature {idx.item():>5}  act={act.item():6.2f}  peak at {tok!r:<14} {url}")
            top_feats.append({"feature": idx.item(), "act": act.item(), "peak_token": tok, "url": url})
        results["prompts"].append({
            "prompt": prompt, "l0": l0, "explained_variance": 1 - fvu,
            "ce_clean": clean_loss, "ce_spliced": spliced_loss, "top_features": top_feats,
        })
        if heat is None:  # token x feature activations for the first prompt's top features
            ids = [f["feature"] for f in top_feats]
            heat = (str_tokens[1:], ids, feats[:, ids].cpu())

    # --- Figure + numbers for the write-up ------------------------------------------------
    out = output_dir("00_setup")
    toks, ids, acts = heat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax1.imshow(acts.T.numpy(), aspect="auto", cmap="Oranges")
    ax1.set(xticks=range(len(toks)), yticks=range(len(ids)), title="top SAE features on prompt 1",
            ylabel="feature")
    ax1.set_xticklabels(toks, rotation=60, fontsize=8)
    ax1.set_yticklabels(ids)
    plt.colorbar(im, ax=ax1, fraction=0.03)
    names = [f"prompt {i + 1}" for i in range(len(results["prompts"]))]
    ev = [r["explained_variance"] for r in results["prompts"]]
    dce = [r["ce_spliced"] - r["ce_clean"] for r in results["prompts"]]
    x = torch.arange(len(names)).numpy()
    ax2.bar(x - 0.2, ev, 0.4, label="explained variance")
    ax2.bar(x + 0.2, dce, 0.4, label="CE loss increase (nats)")
    ax2.set(xticks=x, xticklabels=names, title="reconstruction quality")
    ax2.legend()
    fig.tight_layout()
    fig_path = out / f"sae_{RELEASE}_L{args.layer}.png"
    fig.savefig(fig_path, dpi=150)
    json_path = out / f"sae_{RELEASE}_L{args.layer}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nfigure -> {fig_path}\nnumbers -> {json_path}")


if __name__ == "__main__":
    main()

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

import torch
from sae_lens import SAE, HookedSAETransformer

from ladder import get_device

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
        for act, idx in zip(*peak.topk(args.top_k)):
            tok = str_tokens[1 + pos[idx].item()]
            url = NEURONPEDIA.format(layer=args.layer, idx=idx.item())
            print(f"  feature {idx.item():>5}  act={act.item():6.2f}  peak at {tok!r:<14} {url}")


if __name__ == "__main__":
    main()

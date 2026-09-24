"""Setup 2 — nnsight: read (and edit) activations, locally or on NDIF's hosted models.

TransformerLens needs the weights on your machine; nnsight traces the same
intervention code either locally or remotely on NDIF (https://ndif.us), so you
can poke at models far bigger than your box.

Goals:
  * grab a mid-layer residual stream and the final logits inside a `trace`
  * do a first intervention: zero one layer's MLP output and measure the change
  * run the identical code remotely on a large hosted model

Usage:
  uv run python experiments/00_setup/02_nnsight_activations.py                 # local GPT-2
  uv run python experiments/00_setup/02_nnsight_activations.py --remote        # NDIF (needs NDIF_API_KEY)
  uv run python experiments/00_setup/02_nnsight_activations.py --remote --model meta-llama/Llama-3.1-70B
"""

import argparse
import os

import torch
from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel

PROMPT = "The Eiffel Tower is located in the city of"


def layers(model):
    """Decoder blocks for GPT-2-style (`transformer.h`) or Llama-style (`model.layers`) models."""
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.model.layers


def value(proxy):
    """nnsight >=0.4 returns plain tensors from .save(); older versions wrap them."""
    return getattr(proxy, "value", proxy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--remote", action="store_true", help="run on NDIF instead of locally")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    load_dotenv()
    if args.remote:
        key = os.environ.get("NDIF_API_KEY")
        if not key:
            raise SystemExit("--remote needs NDIF_API_KEY (https://login.ndif.us) in .env")
        CONFIG.set_default_api_key(key)
    model_name = args.model or ("meta-llama/Llama-3.1-8B" if args.remote else "openai-community/gpt2")

    # Remote: weights stay on NDIF, only the meta model is built locally.
    model = LanguageModel(model_name, device_map="auto", dispatch=not args.remote)
    blocks = layers(model)
    mid = len(blocks) // 2
    print(f"{model_name}: {len(blocks)} layers, remote={args.remote}")

    # --- 1. Read activations ------------------------------------------------------
    with model.trace(args.prompt, remote=args.remote):
        out = blocks[mid].output
        # Decoder blocks return a tuple in older transformers, a bare tensor in newer ones.
        hidden = (out[0] if isinstance(out, tuple) else out).save()
        logits = model.lm_head.output.save()
    hidden, logits = value(hidden), value(logits)
    print(f"layer {mid} residual: {tuple(hidden.shape)}")

    probs = logits[0, -1].softmax(-1)
    top = probs.topk(5)
    print(f"prompt: {args.prompt!r}")
    for p, idx in zip(top.values, top.indices):
        print(f"  {model.tokenizer.decode(idx)!r:>12}  {p.item():.3f}")

    # --- 2. Intervene: knock out one MLP ------------------------------------------
    answer = top.indices[0]
    with model.trace(args.prompt, remote=args.remote):
        blocks[mid].mlp.output[:] = 0
        ablated = model.lm_head.output[0, -1].softmax(-1)[answer].save()
    ablated = value(ablated)
    print(f"\nzeroing layer-{mid} MLP: p({model.tokenizer.decode(answer)!r}) "
          f"{probs[answer].item():.3f} -> {float(ablated):.3f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

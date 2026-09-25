"""2.4 (original) — trace the circuit behind a single-token completion of your choice.

This is scaffolding, not an answer: you choose the behaviour, write the prompt pairs, and
decide what to patch next. See 04_trace_circuit.md for the method and the checklist.

A task file is JSON:
  {
    "name": "my_behaviour",
    "model": "gpt2-small",
    "pairs": [
      {"clean": "...", "corrupt": "...", "answer": " X", "wrong": " Y"},
      ...
    ]
  }
All clean prompts must tokenise to the same length, and so must the corrupted ones (pairs are batched).
`answer` / `wrong` are single tokens (note the leading space).

Usage:
  uv run python experiments/01_mech_interp/04_trace_circuit.py TASK.json               # resid + head maps
  uv run python experiments/01_mech_interp/04_trace_circuit.py TASK.json \\
      --senders 9.9 9.6 --receivers logits                                             # path patching
  uv run python experiments/01_mech_interp/04_trace_circuit.py TASK.json \\
      --senders 7.3 7.9 --receivers 9.9q 9.6q                                          # into q/k/v of heads

Sanity check of the tooling on the canonical (already solved) IOI circuit — pick something else
for the card: experiments/01_mech_interp/tasks/ioi_example.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from ladder import get_device, output_dir, patching


def parse_head(s: str) -> tuple[int, int]:
    layer, head = s.split(".")
    return int(layer), int(head)


def parse_receiver(s: str) -> tuple[int, int, str]:
    kind = s[-1]
    if kind not in "qkv":
        raise argparse.ArgumentTypeError(f"receiver {s!r} must end in q, k or v, e.g. 9.9q")
    layer, head = parse_head(s[:-1])
    return layer, head, kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--senders", nargs="*", default=[], help="heads L.H for path patching")
    parser.add_argument("--receivers", nargs="*", default=[],
                        help="'logits' or heads with input kind, e.g. 9.9q 10.0k")
    parser.add_argument("--skip-maps", action="store_true", help="only run the path patching")
    args = parser.parse_args()

    task = json.loads(args.task.read_text())
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(task.get("model", "gpt2-small"), device=get_device())
    pairs = task["pairs"]
    clean = model.to_tokens([p["clean"] for p in pairs])
    corrupt = model.to_tokens([p["corrupt"] for p in pairs])
    if clean.shape != corrupt.shape:
        raise SystemExit(f"clean {tuple(clean.shape)} and corrupt {tuple(corrupt.shape)} token shapes differ")
    answers = torch.tensor([model.to_single_token(p["answer"]) for p in pairs])
    wrongs = torch.tensor([model.to_single_token(p["wrong"]) for p in pairs])
    metric = patching.logit_diff_metric(answers, wrongs)
    cache, norm = patching.baseline(model, clean, corrupt, metric)
    print(f"{task['name']}: {len(pairs)} pairs, logit diff clean {norm.clean:.2f}, corrupt {norm.corrupt:.2f}")

    out = output_dir(f"01_mech_interp/circuit_{task['name']}")
    results = {"task": task["name"], "clean_logit_diff": norm.clean, "corrupt_logit_diff": norm.corrupt}

    if not args.skip_maps:
        resid = patching.patch_resid(model, cache, corrupt, metric, norm)
        heads = patching.patch_heads(model, cache, corrupt, metric, norm)
        results["resid_pre"] = resid.tolist()
        results["heads"] = heads.tolist()
        labels = [f"{i}:{t}" for i, t in enumerate(model.to_str_tokens(clean[0]))]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.6, 1]})
        for ax, data, title in [(ax1, resid, "patch resid_pre at (layer, position)"),
                                (ax2, heads, "patch head output (all positions)")]:
            v = data.abs().max().item()
            im = ax.imshow(data.numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
            ax.set(title=title, ylabel="layer")
            plt.colorbar(im, ax=ax, fraction=0.04)
        ax1.set_xticks(range(len(labels)), labels, rotation=75, fontsize=7)
        ax2.set(xlabel="head", xticks=range(model.cfg.n_heads))
        fig.tight_layout()
        fig.savefig(out / "maps.png", dpi=150)
        flat = heads.flatten()
        top = [(i // model.cfg.n_heads, i % model.cfg.n_heads, flat[i].item()) for i in flat.abs().topk(8).indices.tolist()]
        print("heads with the largest |effect|:", ", ".join(f"{la}.{h} {v:+.2f}" for la, h, v in top))

    if args.senders:
        senders = [parse_head(s) for s in args.senders]
        receivers = None if args.receivers in ([], ["logits"]) else [parse_receiver(r) for r in args.receivers]
        value = patching.path_patch(model, cache, corrupt, metric, norm, senders=senders, receivers=receivers)
        label = "logits" if receivers is None else " ".join(args.receivers)
        print(f"path patching {' '.join(args.senders)} -> {label}: {value:+.3f}")
        results.setdefault("path_patching", []).append({"senders": args.senders, "receivers": label, "value": value})

    path = out / "results.json"
    if path.exists():  # keep earlier path-patching experiments
        old = json.loads(path.read_text())
        results["path_patching"] = old.get("path_patching", []) + results.get("path_patching", [])
        results = {**old, **results}
    path.write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

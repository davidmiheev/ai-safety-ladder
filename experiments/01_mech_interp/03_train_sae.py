"""2.3 — Train your own sparse autoencoder on GPT-2 small and compare it with a pretrained one.

Trains a TopK SAE on blocks.8.hook_resid_pre — the same hook as the pretrained
`gpt2-small-res-jb` SAE from §1 — on OpenWebText, then evaluates both on the same held-out
tokens:
  * L0, explained variance, spliced CE loss (and the zero-ablation CE for scale),
  * dead / dense features,
  * decoder-direction matching: for each of our features, the max cosine similarity to any
    pretrained feature — where do two independently trained dictionaries agree?
  * top-activating contexts for a few of our features, to annotate by hand
    (notes/sae_features_trained.md).

TopK (Gao et al., 2024) instead of the L1 penalty: sparsity is set directly by k, so there's
no λ to tune on a limited GPU budget.

Needs a GPU: roughly an hour for the default 40M tokens on a Colab T4.
  uv run python experiments/01_mech_interp/03_train_sae.py train
  uv run python experiments/01_mech_interp/03_train_sae.py eval            # after train
  uv run python experiments/01_mech_interp/03_train_sae.py eval --replot   # figures from saved JSON
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ladder import get_device, output_dir

DATASET = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
PRETRAINED = ("gpt2-small-res-jb", "blocks.{layer}.hook_resid_pre")


def sae_dir(args) -> Path:
    return output_dir("01_mech_interp") / f"sae_L{args.layer}_topk{args.k}_{args.d_sae}"


def train(args) -> None:
    from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, LoggingConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    steps = args.tokens // args.batch_tokens
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(d_in=768, d_sae=args.d_sae, k=args.k, device=args.device, dtype="float32"),
        model_name="gpt2-small",
        hook_name=f"blocks.{args.layer}.hook_resid_pre",
        dataset_path=DATASET,
        dataset_trust_remote_code=False,  # parquet dataset; silences a datasets warning
        is_dataset_tokenized=True,
        streaming=True,
        context_size=128,
        training_tokens=args.tokens,
        train_batch_size_tokens=args.batch_tokens,
        # Bigger LM batches keep the T4 busy; 32 prompts x 128 tokens gave ~8.6k tok/s.
        store_batch_size_prompts=args.lm_batch,
        # SAELens sizes the shuffle buffer as n_batches_in_buffer * context_size activations; it
        # must hold >= 1 training batch. 256 * 128 = 32k activations = 8 batches of mixing.
        n_batches_in_buffer=256,
        lr=args.lr,
        lr_warm_up_steps=min(1000, steps // 20),
        lr_decay_steps=steps // 5,
        device=args.device,
        seed=args.seed,
        dtype="float32",
        autocast_lm=args.device == "cuda",
        exclude_special_tokens=True,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=0,
        checkpoint_path=None,
        output_path=None,
    )
    t0 = time.time()
    sae = LanguageModelSAETrainingRunner(cfg).run()
    out = sae_dir(args)
    sae.save_inference_model(out)
    (out / "train_info.json").write_text(json.dumps({
        "tokens": args.tokens, "d_sae": args.d_sae, "k": args.k, "layer": args.layer, "lr": args.lr,
        "seed": args.seed, "minutes": (time.time() - t0) / 60, "dataset": DATASET,
    }, indent=2))
    print(f"trained in {(time.time() - t0) / 60:.1f} min -> {out}")


@torch.no_grad()
def evaluate(args) -> dict:
    from datasets import load_dataset
    from sae_lens import SAE, HookedSAETransformer

    device = args.device
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    ours = SAE.load_from_disk(sae_dir(args), device=device)
    release, hook_t = PRETRAINED
    theirs = SAE.from_pretrained(release, hook_t.format(layer=args.layer), device=device)
    hook = f"blocks.{args.layer}.hook_resid_pre"

    # Held-out text: the last shard of the dataset (training streams from the start).
    ds = load_dataset(DATASET, split="train", streaming=True,
                      data_files={"train": "data/train-00072-of-00073.parquet"})
    seqs = []
    for row in ds:
        ids = row["input_ids"]
        for i in range(0, len(ids) - 128, 128):
            seqs.append(ids[i:i + 127])
        if len(seqs) >= args.eval_seqs:
            break
    bos = model.tokenizer.bos_token_id
    tokens = torch.tensor([[bos] + s for s in seqs[:args.eval_seqs]], device=device)

    def zero_hook(x, hook):
        return torch.zeros_like(x)

    stats = {name: {"l0": [], "fvu_num": 0.0, "fvu_den": 0.0, "ce": [], "fires": None}
             for name in ("ours", "pretrained")}
    ce_clean, ce_zero = [], []
    for batch in tokens.split(args.eval_batch):
        _, cache = model.run_with_cache(batch, names_filter=hook)
        x = cache[hook][:, 1:].reshape(-1, 768)  # BOS excluded, as in §1
        ce_clean.append(model(batch, return_type="loss").item())
        ce_zero.append(model.run_with_hooks(batch, return_type="loss", fwd_hooks=[(hook, zero_hook)]).item())
        for name, sae in (("ours", ours), ("pretrained", theirs)):
            f = sae.encode(x)
            xh = sae.decode(f)
            s = stats[name]
            s["l0"].append((f > 0).sum(-1).float().mean().item())
            s["fvu_num"] += (x - xh).pow(2).sum().item()
            s["fvu_den"] += (x - x.mean(0)).pow(2).sum().item()
            s["ce"].append(model.run_with_saes(batch, saes=[sae], return_type="loss").item())
            fires = (f > 0).sum(0)
            s["fires"] = fires if s["fires"] is None else s["fires"] + fires
    n_tokens = tokens.shape[0] * (tokens.shape[1] - 1)

    clean, zero = float(np.mean(ce_clean)), float(np.mean(ce_zero))
    summary = {"n_eval_tokens": n_tokens, "ce_clean": clean, "ce_zero_ablation": zero}
    for name, sae in (("ours", ours), ("pretrained", theirs)):
        s = stats[name]
        freq = (s["fires"].float() / n_tokens).cpu()
        ce = float(np.mean(s["ce"]))
        summary[name] = {
            "d_sae": sae.cfg.d_sae, "l0": float(np.mean(s["l0"])),
            "explained_variance": 1 - s["fvu_num"] / s["fvu_den"],
            "ce_spliced": ce, "ce_loss_recovered": (zero - ce) / (zero - clean),
            "dead_fraction": (freq == 0).float().mean().item(),
            "dense_fraction": (freq > 0.1).float().mean().item(),
            "log10_freq_hist": np.histogram(np.log10(freq[freq > 0].numpy()), bins=40, range=(-7, 0))[0].tolist(),
        }

    # Decoder matching: max cosine of each of our (alive) directions to any pretrained direction.
    alive = (stats["ours"]["fires"] > 0)
    d_ours = torch.nn.functional.normalize(ours.W_dec[alive], dim=-1)
    d_theirs = torch.nn.functional.normalize(theirs.W_dec, dim=-1)
    max_cos = torch.cat([(chunk @ d_theirs.T).max(-1).values for chunk in d_ours.split(1024)]).cpu()
    summary["matching"] = {
        "hist": np.histogram(max_cos.numpy(), bins=40, range=(0, 1))[0].tolist(),
        "median_max_cos": max_cos.median().item(),
        "frac_above_0.7": (max_cos > 0.7).float().mean().item(),
        "frac_above_0.9": (max_cos > 0.9).float().mean().item(),
    }

    # Top-activating contexts for a few alive features of ours, for hand annotation.
    rng = np.random.default_rng(args.seed)
    alive_ids = alive.nonzero().flatten().cpu().numpy()
    picks = rng.choice(alive_ids, size=args.n_annotate, replace=False)
    examples = {int(i): [] for i in picks}
    for bi, batch in enumerate(tokens.split(args.eval_batch)):
        _, cache = model.run_with_cache(batch, names_filter=hook)
        f = ours.encode(cache[hook])[..., picks]  # [b, pos, n]
        for j, fid in enumerate(picks):
            acts = f[..., j]
            vals, flat = acts.flatten().topk(5)
            for v, fl in zip(vals.tolist(), flat.tolist()):
                if v <= 0:
                    continue
                b, p = divmod(fl, acts.shape[1])
                ctx = model.to_string(batch[b, max(1, p - 12):p]) + "⟦" + model.to_string(batch[b, p:p + 1]) + "⟧" \
                    + model.to_string(batch[b, p + 1:p + 4])
                examples[int(fid)].append((v, ctx.replace("\n", "↵")))
    summary["annotate"] = []
    for fid, exs in examples.items():
        exs = sorted(exs, reverse=True)[:6]
        cos_row = torch.nn.functional.normalize(ours.W_dec[fid], dim=-1) @ d_theirs.T
        best = int(cos_row.argmax())
        summary["annotate"].append({
            "feature": fid, "frequency": (stats["ours"]["fires"][fid].item() / n_tokens),
            "closest_pretrained": best, "closest_cos": cos_row[best].item(),
            "neuronpedia_closest": f"https://www.neuronpedia.org/gpt2-small/{args.layer}-res-jb/{best}",
            "top_contexts": [{"act": v, "text": t} for v, t in exs],
        })
    return summary


def plot(summary: dict, out: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    names = ["ours", "pretrained"]
    metrics = [("explained variance", "explained_variance"), ("CE loss recovered", "ce_loss_recovered")]
    x = np.arange(len(metrics))
    for i, n in enumerate(names):
        ax1.bar(x + (i - 0.5) * 0.35, [summary[n][k] for _, k in metrics], 0.35,
                label=f"{n} (L0={summary[n]['l0']:.0f}, d={summary[n]['d_sae']})")
    ax1.set(xticks=x, xticklabels=[m for m, _ in metrics], ylim=(0, 1.05), title="reconstruction quality")
    ax1.legend(fontsize=8, loc="lower right")
    edges = np.linspace(-7, 0, 41)
    for n in names:
        ax2.stairs(summary[n]["log10_freq_hist"], edges, label=n)
    ax2.set(xlabel="log₁₀ firing frequency", ylabel="# features", title="feature density (alive features)")
    ax2.legend()
    ax3.stairs(summary["matching"]["hist"], np.linspace(0, 1, 41), fill=True, color="#3d5a4a", alpha=0.7)
    ax3.axvline(0.7, color="grey", ls=":")
    ax3.set(xlabel="max cosine to any pretrained decoder direction", ylabel="# of our features",
            title=f"do the dictionaries agree? (median {summary['matching']['median_max_cos']:.2f})")
    fig.tight_layout()
    fig.savefig(out / "sae_trained_vs_pretrained.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--d-sae", type=int, default=12288)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--tokens", type=int, default=40_000_000)
    parser.add_argument("--lm-batch", type=int, default=128, help="prompts per LM forward when harvesting")
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seqs", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--n-annotate", type=int, default=8)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()

    out = output_dir("01_mech_interp")
    if args.mode == "train":
        train(args)
        return
    path = out / "sae_trained_vs_pretrained.json"
    summary = json.loads(path.read_text()) if args.replot else evaluate(args)
    if not args.replot:
        path.write_text(json.dumps(summary, indent=1))
        for n in ("ours", "pretrained"):
            s = summary[n]
            print(f"{n:<10} L0 {s['l0']:.1f}  EV {s['explained_variance']:.3f}  CE {s['ce_spliced']:.3f} "
                  f"(recovered {s['ce_loss_recovered']:.3f})  dead {s['dead_fraction']:.3f}")
        print(f"matching: median max-cos {summary['matching']['median_max_cos']:.2f}, "
              f">0.7: {summary['matching']['frac_above_0.7']:.2f}")
    plot(summary, out)
    print(f"figure -> {out / 'sae_trained_vs_pretrained.png'}\nnumbers -> {path}")


if __name__ == "__main__":
    main()

"""3.1 follow-up — how many frequencies do the modular-addition logits actually need?

Loads the model saved by 01_modular_addition.py and, on all p² inputs:
  * keeps the top-K frequencies of each logit vector over answers c (ranked by mean power) and
    reports loss / accuracy / power share for a range of K;
  * tests whether the top-20 logit frequencies are enriched for the embedding's key frequencies
    and their pairwise sums/differences (combination tones a ReLU MLP would create), against
    random sets of 20 frequencies.

  uv run python experiments/02_math/01b_output_frequencies.py
"""

import importlib.util
import json
import random
from pathlib import Path

import torch

from ladder import output_dir

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("modadd", HERE / "01_modular_addition.py")
modadd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modadd)


def main() -> None:
    out = output_dir("02_math")
    res = json.loads((out / "modular_addition.json").read_text())
    p, key = res["p"], res["key_freqs"]
    torch.set_grad_enabled(False)
    model = modadd.make_model(p, "cpu", res["seed"])
    model.load_state_dict(torch.load(out / "modular_addition_model.pt", map_location="cpu"))

    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    tokens = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], dim=1)
    labels = ((a + b) % p).flatten()
    logits = model(tokens)[:, -1].double()
    power = modadd.logit_spectrum(logits, p)
    order = (power.argsort(descending=True) + 1).tolist()

    def keep(freqs):
        F = torch.fft.rfft(logits, dim=-1)
        mask = torch.zeros(F.shape[-1], dtype=torch.bool)
        mask[freqs] = True
        mask[0] = True
        R = torch.fft.irfft(F * mask, n=p, dim=-1)
        return (torch.nn.functional.cross_entropy(R, labels).item(),
                (R.argmax(-1) == labels).double().mean().item())

    sweep = []
    for k in (1, 3, 5, 10, 20, 30, p // 2):
        loss, acc = keep(order[:k])
        share = (power[[f - 1 for f in order[:k]]].sum() / power.sum()).item()
        sweep.append({"k": k, "loss": loss, "accuracy": acc, "power_share": share})
        print(f"top-{k:>2} logit frequencies: loss {loss:.3f}  acc {acc:.3f}  power share {share:.3f}")

    def fold(k):
        k %= p
        return min(k, p - k)

    combos = {fold(k) for k in key} | {fold(x + y) for x in key for y in key} \
        | {fold(x - y) for x in key for y in key if x != y}
    combos.discard(0)
    top20 = order[:20]
    hits = [k for k in top20 if k in combos]
    rng = random.Random(0)
    null = [sum(k in combos for k in rng.sample(range(1, p // 2 + 1), 20)) for _ in range(5000)]
    null_mean = sum(null) / len(null)
    p_value = sum(n >= len(hits) for n in null) / len(null)
    low_share = (power[:3].sum() / power.sum()).item()
    print(f"top-20 logit freqs in keys ∪ pairwise sums/differences: {len(hits)} "
          f"(random: {null_mean:.1f}, P(≥{len(hits)}) = {p_value:.3f}); k=1..3 power share {low_share:.3f}")

    result = {"sweep": sweep, "top20": top20, "combination_set_size": len(combos), "hits": hits,
              "null_mean_hits": null_mean, "p_value": p_value,
              "hits_power_share": (power[[k - 1 for k in hits]].sum() / power.sum()).item(),
              "low_freq_power_share": low_share, "key_freqs": key}
    (out / "modular_addition_output.json").write_text(json.dumps(result, indent=1))
    print(f"-> {out / 'modular_addition_output.json'}")


if __name__ == "__main__":
    main()

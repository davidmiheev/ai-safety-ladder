"""Activation- and path-patching utilities for TransformerLens HookedTransformers.

The workflow for tracing a circuit (card 2.4):
  1. Pick a clean/corrupted prompt pair that differ only in what the behaviour depends on,
     and a scalar metric (usually the logit difference between the right and wrong answer).
  2. `patch_resid` — patch the clean residual stream into the corrupted run at each
     (layer, position): shows *where* in the forward pass the information lives.
  3. `patch_heads` — patch each head's output at all positions: *which* heads matter.
  4. `path_patch` — patch a sender head's clean output *only along its path to* chosen
     receivers (other heads' q/k/v, or the logits): *how* the heads are wired together.

All functions report the normalised metric (M_patched - M_corrupt) / (M_clean - M_corrupt), so
1 = fully restores clean behaviour and 0 = no effect (negative values are real: the component
pushes against the answer).
"""

from collections.abc import Callable

import torch
from transformer_lens import ActivationCache, HookedTransformer

Metric = Callable[[torch.Tensor], torch.Tensor]  # logits -> scalar tensor


def logit_diff_metric(answer_ids: torch.Tensor, wrong_ids: torch.Tensor, pos: int = -1) -> Metric:
    """Mean over the batch of logit[answer] - logit[wrong] at position `pos`.

    answer_ids, wrong_ids: [batch] token ids (one per prompt).
    """
    def metric(logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, pos]
        rows = torch.arange(last.shape[0], device=last.device)
        return (last[rows, answer_ids.to(last.device)] - last[rows, wrong_ids.to(last.device)]).mean()
    return metric


class Normaliser:
    """Maps raw metric values to (M - M_corrupt) / (M_clean - M_corrupt)."""

    def __init__(self, clean: float, corrupt: float):
        if abs(clean - corrupt) < 1e-6:
            raise ValueError("clean and corrupted metric are equal: the prompt pair doesn't isolate the behaviour")
        self.clean, self.corrupt = clean, corrupt

    def __call__(self, value: float) -> float:
        return (value - self.corrupt) / (self.clean - self.corrupt)


def baseline(model: HookedTransformer, clean: torch.Tensor, corrupt: torch.Tensor, metric: Metric):
    """Run both prompts once. Returns (clean_cache, normaliser)."""
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits = model(corrupt)
    norm = Normaliser(metric(clean_logits).item(), metric(corrupt_logits).item())
    return clean_cache, norm


@torch.no_grad()
def patch_resid(model: HookedTransformer, clean_cache: ActivationCache, corrupt: torch.Tensor,
                metric: Metric, norm: Normaliser, hook: str = "resid_pre") -> torch.Tensor:
    """[n_layers, seq] map: patch clean `hook` (resid_pre / attn_out / mlp_out) at one (layer, pos)."""
    n_layers, seq = model.cfg.n_layers, corrupt.shape[1]
    out = torch.zeros(n_layers, seq)
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_{hook}"
        for pos in range(seq):
            def fn(x, hook, pos=pos, name=name):
                x[:, pos] = clean_cache[name][:, pos]
                return x
            out[layer, pos] = norm(metric(model.run_with_hooks(corrupt, fwd_hooks=[(name, fn)])).item())
    return out


@torch.no_grad()
def patch_heads(model: HookedTransformer, clean_cache: ActivationCache, corrupt: torch.Tensor,
                metric: Metric, norm: Normaliser, pos: int | None = None) -> torch.Tensor:
    """[n_layers, n_heads] map: patch one head's clean output (hook_z), at all positions or one."""
    cfg = model.cfg
    out = torch.zeros(cfg.n_layers, cfg.n_heads)
    sl = slice(None) if pos is None else pos
    for layer in range(cfg.n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        for head in range(cfg.n_heads):
            def fn(z, hook, head=head, name=name):
                z[:, sl, head] = clean_cache[name][:, sl, head]
                return z
            out[layer, head] = norm(metric(model.run_with_hooks(corrupt, fwd_hooks=[(name, fn)])).item())
    return out


@torch.no_grad()
def path_patch(model: HookedTransformer, clean_cache: ActivationCache, corrupt: torch.Tensor,
               metric: Metric, norm: Normaliser, senders: list[tuple[int, int]],
               receivers: list[tuple[int, int, str]] | None = None) -> float:
    """Effect of `senders`' clean outputs that flows *only* through `receivers`.

    senders:   [(layer, head)] whose output (hook_z) is taken from the clean run.
    receivers: [(layer, head, "q"|"k"|"v")] whose input is allowed to see the change; None means
               the direct path to the logits (the final residual stream).

    Three passes (as in Wang et al. 2022 / ARENA):
      1. corrupted run with senders patched to clean and *every other head frozen* at its
         corrupted output. MLPs recompute, as in the standard formulation, so a path
         sender -> MLP -> receiver counts as part of the sender's path;
      2. record the receivers' inputs (or the final residual) from that run;
      3. corrupted run with only those receiver inputs replaced.
    """
    cfg = model.cfg
    _, corrupt_cache = model.run_with_cache(corrupt)
    sender_set = set(senders)

    def freeze_or_send(layer: int):
        name = f"blocks.{layer}.attn.hook_z"

        def fn(z, hook):
            for head in range(cfg.n_heads):
                src = clean_cache if (layer, head) in sender_set else corrupt_cache
                z[:, :, head] = src[name][:, :, head]
            return z
        return name, fn

    hooks = [freeze_or_send(layer) for layer in range(cfg.n_layers)]
    if receivers is None:
        final = f"blocks.{cfg.n_layers - 1}.hook_resid_post"
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(corrupt, names_filter=final)
        patched = [(final, lambda x, hook: cache[final])]
    else:
        names = sorted({f"blocks.{layer}.attn.hook_{kind}" for layer, _, kind in receivers})
        # Receivers read their input before their own output, so freezing their outputs is fine.
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(corrupt, names_filter=lambda n: n in names)
        by_name: dict[str, list[int]] = {}
        for layer, head, kind in receivers:
            by_name.setdefault(f"blocks.{layer}.attn.hook_{kind}", []).append(head)

        def recv(name, heads):
            def fn(x, hook):
                x[:, :, heads] = cache[name][:, :, heads]
                return x
            return fn
        patched = [(name, recv(name, heads)) for name, heads in by_name.items()]
    return norm(metric(model.run_with_hooks(corrupt, fwd_hooks=patched)).item())

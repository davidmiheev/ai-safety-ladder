"""Invariants of ladder.patching on GPT-2 small (downloads weights; marked slow)."""

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup():
    from transformer_lens import HookedTransformer

    from ladder import patching

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean = model.to_tokens(["When Mary and John went to the store, John gave a drink to"])
    corrupt = model.to_tokens(["When Mary and John went to the store, Mary gave a drink to"])
    diff_pos = int((clean != corrupt).nonzero()[0, 1])
    metric = patching.logit_diff_metric(
        torch.tensor([model.to_single_token(" Mary")]), torch.tensor([model.to_single_token(" John")]))
    cache, norm = patching.baseline(model, clean, corrupt, metric)
    return model, patching, clean, corrupt, diff_pos, metric, cache, norm


def test_resid_layer0_localises_the_edit(setup):
    model, patching, _, corrupt, diff_pos, metric, cache, norm = setup
    m = patching.patch_resid(model, cache, corrupt, metric, norm)
    # Layer-0 residual differs only at the edited token: patching it there restores everything,
    # patching anywhere else changes nothing.
    assert m[0, diff_pos] == pytest.approx(1.0, abs=1e-4)
    others = torch.cat([m[0, :diff_pos], m[0, diff_pos + 1:]])
    assert others.abs().max() < 1e-4
    # resid_pre at the last layer is *not* exactly 1 at the final position: layer 11's attention
    # there still reads corrupted residuals at other positions. resid_post of the last layer is.
    assert m[-1, -1] < 1.0
    post = patching.patch_resid(model, cache, corrupt, metric, norm, hook="resid_post")
    assert post[-1, -1] == pytest.approx(1.0, abs=1e-4)


def test_path_patch_all_heads_to_logits_equals_patching_all_heads(setup):
    model, patching, _, corrupt, _, metric, cache, norm = setup
    cfg = model.cfg
    everyone = [(la, h) for la in range(cfg.n_layers) for h in range(cfg.n_heads)]
    via_paths = patching.path_patch(model, cache, corrupt, metric, norm, senders=everyone)
    hooks = [(f"blocks.{la}.attn.hook_z", lambda z, hook: cache[hook.name]) for la in range(cfg.n_layers)]
    direct = norm(metric(model.run_with_hooks(corrupt, fwd_hooks=hooks)).item())
    assert via_paths == pytest.approx(direct, abs=1e-4)


def test_path_patch_to_receiver_only_moves_that_receiver(setup):
    model, patching, _, corrupt, _, metric, cache, norm = setup
    # A receiver in layer 0 can't be reached by a sender in layer 5: no effect at all.
    assert patching.path_patch(model, cache, corrupt, metric, norm, senders=[(5, 5)],
                               receivers=[(0, 0, "q")]) == pytest.approx(0.0, abs=1e-6)


def test_ioi_name_movers_matter_most_on_direct_path(setup):
    model, patching, _, corrupt, _, metric, cache, norm = setup
    heads = patching.patch_heads(model, cache, corrupt, metric, norm)
    top = {(i // 12, i % 12) for i in heads.flatten().topk(5).indices.tolist()}
    # Known IOI name movers (Wang et al., 2022): 9.9, 9.6, 10.0.
    assert top & {(9, 9), (9, 6), (10, 0)}, top

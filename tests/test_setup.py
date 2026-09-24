import pytest
import torch

from ladder import get_device


def test_device_is_usable():
    assert torch.zeros(1, device=get_device()).item() == 0


def test_output_dir_created(tmp_path, monkeypatch):
    from ladder import utils

    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path)
    assert utils.output_dir("x").is_dir()


def test_libraries_import():
    import circuitsvis  # noqa: F401
    import nnsight  # noqa: F401
    from sae_lens import SAE  # noqa: F401
    from transformer_lens import HookedTransformer  # noqa: F401


@pytest.mark.slow
def test_cache_reproduces_logits():
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("pythia-14m", device="cpu")
    logits, cache = model.run_with_cache("Hello world")
    rebuilt = model.unembed(model.ln_final(cache["resid_post", model.cfg.n_layers - 1]))
    assert torch.allclose(rebuilt, logits, atol=1e-4)

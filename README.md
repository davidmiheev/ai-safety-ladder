# AI Safety & Interpretability — Experiments Ladder

Hands-on companion to the reading path. Setup for tooling, reproductions for the
load-bearing results, original work for the questions nobody has answered yet.

- Ladder (interactive checklist): [`docs/experiments-ladder.html`](docs/experiments-ladder.html)
- Reading path: [`docs/reading-list.html`](docs/reading-list.html)
- Write-ups (PDF, one per section, with the math, figures and findings): [`writeups/pdf/`](writeups/pdf/)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10–3.12.

```bash
uv sync --extra cpu            # CPU / Apple Silicon (MPS)
# uv sync --extra cu128        # NVIDIA GPU (CUDA 12.8)
# add --extra notebooks for JupyterLab

cp .env.example .env           # fill in NDIF_API_KEY / HF_TOKEN when you need them
uv run pytest -m "not slow"    # fast sanity check
```

Then work through `experiments/00_setup/` in order:

```bash
uv run python experiments/00_setup/01_transformerlens_first_pass.py
uv run python experiments/00_setup/02_nnsight_activations.py            # local
uv run python experiments/00_setup/02_nnsight_activations.py --remote   # NDIF, needs key
uv run python experiments/00_setup/03_saelens_pretrained_sae.py
```

Generated plots/HTML/JSON land in `outputs/<experiment>/` (git-ignored).

### Write-ups

Each section has a Typst source in `writeups/` that pulls its numbers straight from the
experiments' JSON output, so text and results can't drift apart. After re-running experiments:

```bash
uv run python scripts/build_writeups.py        # copies figures/JSON into writeups/, renders writeups/pdf/*.pdf
uv run python scripts/build_writeups.py 02     # one section
```

## Remote GPUs (Colab / Kaggle via MCP)

Everything in `00_setup` runs on CPU. From section 2 on (training SAEs, sleeper-agent
fine-tuning) you'll want a GPU. The Colab and Kaggle MCP servers come from
[ds-agent](https://github.com/davidmiheev/ds-agent) and are declared in `.mcp.json`:

| Server | Use it for |
|--------|------------|
| `colab` | `colab_new` (CPU / T4 / L4 / A100 / TPU) → `colab_install` → `colab_upload` → `colab_execute` → `colab_stop` |
| `kaggle` | Kaggle datasets, models and notebooks (Kaggle's official MCP, with ds-agent's argument fixes) |

```bash
bash scripts/setup_gpu_mcp.sh      # clones ds-agent into external/, builds both venvs
```

Credentials (never commit them):
- **Colab**: run `external/ds-agent/src/colab_mcp/.venv/bin/python external/ds-agent/src/colab_mcp/auth_once.py`
  on any machine with a browser. It writes `~/.config/colab-cli/token.json`. On a fresh machine, pass
  its contents as `COLAB_TOKEN_JSON` to `setup_gpu_mcp.sh`, or call the `colab_auth` tool (URL, then code).
- **Kaggle**: create an API token (kaggle.com → Settings → API, `KGAT…`) and export it as `KAGGLE_API_TOKEN`
  before starting Claude Code.

Gotchas (verified 2026-09):
- `colab_upload` writes relative to `/`, not `/content` as its description says, and fails for
  sub-paths whose parent doesn't exist there. Upload flat files, then `shutil.move` them in `colab_execute`.
- The Kaggle server launches through `scripts/kaggle_mcp.py`, which passes `HTTPS_PROXY` /
  `NODE_EXTRA_CA_CERTS` through to `npx mcp-remote` (ds-agent's proxy drops them, which breaks
  npm behind TLS-intercepting proxies).
- Release runtimes with `colab_stop` when idle; they burn compute units.

For cloud sessions, add `COLAB_TOKEN_JSON` and `KAGGLE_API_TOKEN` as environment secrets. The
SessionStart hook (`.claude/settings.json` → `scripts/session_start.sh`) runs `uv sync` and
`setup_gpu_mcp.sh` on every fresh cloud container (~17s cold, ~2s warm); it is a no-op locally.

## Layout

```
docs/                  the ladder + reading list (HTML)
src/ladder/            shared helpers (device selection, output paths)
experiments/
  00_setup/            §1 Setup & Tooling
  01_mech_interp/      §2 Mechanistic Interpretability — Reproductions
  02_math/             §3 Math for AI Safety — Worked Problems
  03_oversight/        §4 Scalable Oversight & Behavioral Alignment
  04_reward_hacking/   §5 Reward Hacking & Specification Gaming
  05_synthesis/        §6 Governance & Synthesis
notes/                 your notes: feature annotations, null results
writeups/              section write-ups (Typst sources, figures, data, rendered pdf/)
scripts/               GPU MCP setup (Colab / Kaggle via ds-agent)
tests/                 smoke tests for the toolchain
```

## Progress

### 1 · Setup & Tooling
- [x] TransformerLens + CircuitsVis: first forward pass with activation caching — `00_setup/01_transformerlens_first_pass.py`
- [ ] nnsight: remote activation access on a larger hosted model — `00_setup/02_nnsight_activations.py`
- [ ] SAELens + Neuronpedia: load a pretrained GPT-2 small SAE, inspect 5 features by hand — `00_setup/03_saelens_pretrained_sae.py`, `notes/sae_features_pretrained.md`
- [ ] YSDA NLP course week 11: probing, then mech-interp lab — `00_setup/04_ysda_interpretability.md`

### 2 · Mechanistic Interpretability — Reproductions
- [x] Induction head in GPT-2 small via attention patterns + activation patching (Olsson et al., 2022) — `01_mech_interp/01_induction_heads.py`
- [x] Toy ReLU model: superposition phase-change plot (Elhage et al., 2022) — `01_mech_interp/02_toy_superposition.py`
- [ ] Train your own SAE on one GPT-2 small layer; interpret 5 features — `01_mech_interp/03_train_sae.py` (GPU), `notes/sae_features_trained.md`
- [ ] *Original:* hand-trace a single-token completion circuit with activation/path patching — scaffold + method: `01_mech_interp/04_trace_circuit.{py,md}`, toolkit `src/ladder/patching.py`

### 3 · Math for AI Safety — Worked Problems
- [ ] Modular addition: recover the Fourier embedding structure
- [ ] *Original:* FairBot / CUPOD / DUPOC via toy Löbian proof search
- [ ] *Original:* no-coincidence mini heuristic-estimator argument

### 4 · Scalable Oversight & Behavioral Alignment
- [ ] Model-written sycophancy evals (Perez et al.)
- [ ] Sleeper-agent-lite: does a backdoor survive safety fine-tuning? (Hubinger et al.)
- [ ] Linear probe detecting the backdoor before it fires

### 5 · Reward Hacking & Specification Gaming
- [ ] Gameable-reward RL environment; watch the agent find the exploit
- [ ] *Original:* alignment-faking-lite compliance gap

### 6 · Governance & Synthesis
- [ ] *Original:* map 3 mech-interp papers onto Sharkey et al.'s open-problems taxonomy
- [ ] *Original:* mini safety-vs-capabilities publication tally

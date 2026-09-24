# Setup 4 — YSDA NLP course, week 11: interpretability

Course material: https://github.com/yandexdataschool/nlp_course/tree/2025/week11_interpretability

Do probing first (the simpler technique), then the mechanistic-interpretability lab
as a warm-up for the activation-patching reproductions in section 2.

## Getting the notebooks

```bash
git clone --depth 1 --branch 2025 --filter=blob:none --sparse \
    https://github.com/yandexdataschool/nlp_course external/nlp_course
git -C external/nlp_course sparse-checkout set week11_interpretability
uv sync --extra notebooks
uv run jupyter lab external/nlp_course/week11_interpretability
```

`external/` is git-ignored. Save your completed notebooks here
(`experiments/00_setup/`) so they are tracked.

Notebooks that need a GPU can go to Colab via the `colab` MCP server (see the README).

## Checklist
- [ ] Probing classifiers: train linear probes on layer activations, compare layers
- [ ] Control task / selectivity: is the probe learning the property or memorising?
- [ ] Mech-interp lab: complete every graded cell
- [ ] Notes: where do probing and patching disagree, and why?

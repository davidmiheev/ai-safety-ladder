#import "template.typ": *

#let fp = json("data/first_pass_gpt2-small.json")
#let ns = json("data/nnsight_gpt2.json")
#let sae = json("data/sae_gpt2-small-res-jb_L8.json")

#show: writeup.with(
  section: "§1",
  title: "Setup & Tooling",
  summary: [
    The three tools the rest of the ladder assumes are working and cross-checked.
    *TransformerLens* caches all #fp.n_cached_activations activations of GPT-2 small, and
    the final logits can be rebuilt exactly from the cached residual stream
    (max error #fx(fp.resid_reconstruction_max_err, d: 1)). *nnsight* reads and edits the same
    activations through a model-agnostic tracing API that can also run remotely on NDIF.
    *SAELens* loads a pretrained sparse autoencoder whose features are readable at a glance
    (" Golden", " San", " Francisco"). It explains #pct(sae.prompts.at(0).explained_variance)
    of the residual variance on prose but only #pct(sae.prompts.at(1).explained_variance)
    on code.
  ],
)

= Environment

Everything runs from one `uv` project (Python 3.11, PyTorch 2.14, TransformerLens 3.9,
SAELens 6.51, nnsight 0.7). TransformerLens is pinned below 4.0: version 4 removes
`HookedTransformer`, which ARENA, YSDA and SAELens still use. All of §1 runs on a laptop
CPU in under a minute per script. Section 2 onwards can offload to Colab or Kaggle GPUs
through the MCP servers set up by `scripts/setup_gpu_mcp.sh` (see the README). As a
cross-check, the SAE script gave identical numbers on a Colab T4.

#cmd("uv sync\nuv run python experiments/00_setup/01_transformerlens_first_pass.py\nuv run python experiments/00_setup/02_nnsight_activations.py\nuv run python experiments/00_setup/03_saelens_pretrained_sae.py\nuv run python scripts/build_writeups.py 01")

= TransformerLens: the residual stream

A transformer with $L$ layers keeps a *residual stream* $x_ell in RR^(d_"model")$ at every
position. Each attention layer and MLP reads from it and adds its output back:
$
  x_0 = W_E t + W_"pos", quad
  x_(ell+1) = x_ell + sum_(h=1)^(H) "attn"_(ell,h)(x_ell) + "mlp"_ell (x_ell + sum_h "attn"_(ell,h)(x_ell)),
$
and the logits are $W_U "LN"(x_L)$. Because the stream is a sum, the output of every
component lives in the same space, which is what makes it possible to attribute behaviour
to individual heads.

*Sanity check.* On the IOI prompt _"#fp.prompt"_ we run GPT-2 small
($L = #fp.n_layers$, $H = #fp.n_heads$, $d_"model" = #fp.d_model$) once with
`run_with_cache`. Recomputing $W_U "LN"(x_L)$ from the cached `resid_post` of the last
layer reproduces the forward-pass logits to within #fx(fp.resid_reconstruction_max_err, d: 1).
The model predicts
#fp.top_predictions.slice(0, 3).map(p => [#raw(p.token) (#fx(p.prob))]).join(", ").

*Logit lens.* Applying the unembedding early, $"logits"_ell = W_U "LN"(x_ell)$, shows when
the answer becomes linearly readable. The rank of " Mary" falls from
#fp.logit_lens_rank.at("0_pre") at the embeddings to #fp.logit_lens_rank.at("8_pre") before
layer 8 and #fp.logit_lens_rank.at("9_pre") before layer 9. It reaches rank 0 (top
prediction) before layer 10. This matches the IOI circuit (Wang et al., 2022), whose name-mover
heads sit in layers 9–10. Head #fp.static_head, one such name mover, puts
#fx(fp.static_head_final_query_top_weight) of its final-position attention on
#raw(fp.static_head_final_query_top_key) (@fig-lens, right).

#figure(
  image("figures/logit_lens_gpt2-small.png", width: 100%),
  caption: [Left: logit-lens probability at the final position. Before layer 9 the model
    backs the generic " them" (#fx(fp.logit_lens_prob.at(" them").at(9)) after layer 8). The
    indirect object " Mary" takes over only after layers 9–10, and the subject " John" is
    kept near zero throughout. Right: attention pattern of head #fp.static_head. The final
    query (" to") attends to " Mary".],
) <fig-lens>

= nnsight: tracing and intervening

nnsight wraps a HuggingFace model and records a computation graph inside a
`with model.trace(prompt):` block. Reading `.output` of any module and calling `.save()`
returns the tensor. *Assigning* to it edits the forward pass. The same code runs remotely on
NDIF with `remote=True`, where only the saved tensors are returned. That is the route to
activation access on models too large for local hardware.

On _"#ns.prompt"_, GPT-2 puts #fx(ns.p_answer_clean, d: 3) on " Paris". Zero-ablating the
layer-#ns.ablated_layer MLP (setting its output to $0$) *raises* this to
#fx(ns.p_answer_mlp_zeroed, d: 3).

#caveat[
  Zero-ablation moves activations off-distribution, because an MLP output of exactly $0$
  never occurs in training. A single-layer ablation that "helps" is therefore weak
  evidence. It could be removing an inhibitory contribution, or an artefact of the edit
  itself. Mean-ablation, or resample-ablation from other prompts, are the better
  defaults. Both appear in §2.
]

= SAELens: a pretrained sparse autoencoder

A sparse autoencoder (SAE) rewrites an activation $x in RR^(768)$ as a sparse, non-negative
combination of $d_"sae" = #sae.d_sae$ learned directions:
$
  f(x) = "ReLU"(W_"enc" (x - b_"dec") + b_"enc"), quad
  hat(x) = W_"dec" f(x) + b_"dec",
$
trained to minimise $norm(x - hat(x))_2^2 + lambda norm(f(x))_1$. The hope is that each
coordinate of $f$ is a *monosemantic feature*. We use #raw(sae.release) at
#raw(sae.hook) and report three quality metrics:
- *sparsity* $L_0 = EE norm(f(x))_0$, the number of active features per token;
- *explained variance* $1 - "FVU"$, where
  $"FVU" = sum_t norm(x_t - hat(x)_t)^2 \/ sum_t norm(x_t - macron(x))^2$;
- *spliced CE loss*: next-token loss with $hat(x)$ substituted for $x$ in the forward pass.
  This is the metric that matters, since a reconstruction can have low MSE and still
  destroy information the model uses.

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, right, right, right),
    table.header([], [prompt], [$L_0$], [expl. var.], [CE clean → spliced]),
    ..sae.prompts.enumerate().map(((i, p)) => (
      [#(i + 1)], [#p.prompt.replace("\n", " ↵ ")], [#fx(p.l0, d: 1)],
      [#fx(p.explained_variance, d: 3)], [#fx(p.ce_clean) → #fx(p.ce_spliced)],
    )).flatten(),
  ),
  caption: [Reconstruction quality of the layer-8 residual SAE on three prompts (BOS excluded).],
) <tab-sae>

#figure(
  image("figures/sae_gpt2-small-res-jb_L8.png", width: 100%),
  caption: [Left: activations of prompt 1's five strongest features across its tokens. Each
    fires on essentially one token. Right: explained variance and CE-loss increase per prompt.],
) <fig-sae>

#finding[
  On prose the SAE is good: $L_0 approx #fx(sae.prompts.at(0).l0, d: 0)$ out of #sae.d_sae,
  #pct(sae.prompts.at(0).explained_variance) variance explained, and splicing it in costs
  #fx(sae.prompts.at(0).ce_spliced - sae.prompts.at(0).ce_clean) nats. On code it is
  markedly worse (#pct(sae.prompts.at(1).explained_variance) explained,
  +#fx(sae.prompts.at(1).ce_spliced - sae.prompts.at(1).ce_clean) nats), which suggests
  code is under-represented in its training distribution (OpenWebText). The strongest
  feature on the code prompt fires on a whitespace token, most likely indentation.
]

The top features on prompt 1 fire on exactly the tokens you would guess:
#sae.prompts.at(0).top_features.slice(0, 4).map(f =>
  link(f.url)[\##f.feature] + [ (#raw(f.peak_token))]).join(", ").
Hand-annotating five features, with a hypothesis formed *before* reading Neuronpedia's
label and then stress-tested with your own prompts, is part of card 1.3 and lives in
`notes/sae_features_pretrained.md`.

= Status of §1

#status-table(
  (true, [TransformerLens + CircuitsVis first pass, logit lens, residual reconstruction.]),
  ("partial", [nnsight: local tracing and intervention done. Remote NDIF run pending an API key.]),
  ("partial", [SAELens: loading, metrics and feature listing done. Hand-annotation of 5 features (your notes) to do.]),
  (false, [YSDA week 11: probing, then the mech-interp lab (`experiments/00_setup/04_ysda_interpretability.md`).]),
)

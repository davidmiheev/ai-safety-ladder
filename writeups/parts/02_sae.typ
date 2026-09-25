#import "../template.typ": *
#let S = json("../data/sae_trained_vs_pretrained.json")
#let T = json("../data/sae_train_info.json")
#let O = S.ours
#let P = S.pretrained

Card 2.3: train a sparse autoencoder from scratch and compare it with a pretrained one.
Code: `experiments/01_mech_interp/03_train_sae.py`. Training ran on a Colab T4 through the MCP
server (#fx(T.minutes, d: 0) min), and so did the evaluation. Hand annotation of five features
is your part of the card: `notes/sae_features_trained.md`.

== Setup

Same hook as the pretrained SAE of §1 (`blocks.#T.layer.hook_resid_pre`), same data
distribution (OpenWebText, 128-token contexts), #fx(T.tokens / 1e6, d: 0)M training tokens.
Instead of the L1 penalty of §1 we use a *TopK* SAE (Gao et al., 2024):
$
  f(x) = "TopK"_k ("ReLU"(W_"enc" (x - b_"dec") + b_"enc")), quad
  hat(x) = W_"dec" f(x) + b_"dec", quad
  cal(L) = norm(x - hat(x))_2^2 + alpha cal(L)_"aux",
$
where $"TopK"_k$ keeps the $k = #T.k$ largest pre-activations and zeroes the rest. Sparsity is
set directly, so there's no $lambda$ to tune on a small compute budget. $cal(L)_"aux"$ makes dead
latents reconstruct the residual error, which keeps them alive. Dictionary size
$d_"sae" = #T.d_sae$ ($16 times$), half the pretrained SAE's #P.d_sae.

On #S.n_eval_tokens held-out tokens from the last dataset shard, both SAEs get the same metrics
as §1, plus the fraction of CE loss recovered relative to zero-ablating the layer:
$
  "CE recovered" = (cal(L)_"zero" - cal(L)_"SAE") / (cal(L)_"zero" - cal(L)_"clean").
$
To ask whether two independently trained dictionaries found the *same* features, match each of
our live decoder directions $d_i$ to its nearest pretrained direction:
$"match"(i) = max_j cos(d_i, d'_j)$.

== Results

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.header([], [ours (TopK)], [pretrained (L1)]),
    [dictionary size $d_"sae"$], [#O.d_sae], [#P.d_sae],
    [$L_0$], [#fx(O.l0, d: 1)], [#fx(P.l0, d: 1)],
    [explained variance], [#fx(O.explained_variance, d: 3)], [#fx(P.explained_variance, d: 3)],
    [spliced CE (clean #fx(S.ce_clean, d: 3), zero-abl. #fx(S.ce_zero_ablation, d: 2))], [#fx(O.ce_spliced, d: 3)], [#fx(P.ce_spliced, d: 3)],
    [CE loss recovered], [#fx(O.ce_loss_recovered, d: 3)], [#fx(P.ce_loss_recovered, d: 3)],
    [dead features (never fired)], [#pct(O.dead_fraction, d: 1)], [#pct(P.dead_fraction, d: 1)],
    [dense features (fire > 10%)], [#pct(O.dense_fraction, d: 2)], [#pct(P.dense_fraction, d: 2)],
  ),
  caption: [Our TopK SAE against the pretrained SAE on the same held-out tokens.],
) <tab-sae-cmp>

#figure(
  image("../figures/sae_trained_vs_pretrained.png", width: 100%),
  caption: [Left: reconstruction quality. Middle: distribution of feature firing frequencies.
    Right: for each of our live features, the cosine similarity of its decoder direction to the
    closest pretrained one.],
) <fig-sae-cmp>

#let M = S.matching
#finding[
  *Agreement between dictionaries:* the median of our features' best cosine to a pretrained
  feature is #fx(M.median_max_cos); #pct(M.at("frac_above_0.7")) of them have a pretrained
  counterpart above $0.7$ and #pct(M.at("frac_above_0.9")) above $0.9$.
]

== Features to annotate

A random sample of our live features, with their strongest held-out contexts (the active token
in ⟦brackets⟧) and their closest pretrained feature. Form a hypothesis *before* opening the
Neuronpedia link.

#for a in S.annotate [
  #block(breakable: false, above: 10pt)[
    *Feature #a.feature* #h(4pt) #text(size: 8.5pt, fill: soft)[fires on #pct(a.frequency, d: 3) of tokens ·
    closest pretrained #link(a.neuronpedia_closest)[\##a.closest_pretrained] (cos #fx(a.closest_cos))]
    #set text(size: 8.5pt)
    #for c in a.top_contexts.slice(0, calc.min(4, a.top_contexts.len())) [
      - #fx(c.act, d: 1): #raw(c.text)
    ]
  ]
]

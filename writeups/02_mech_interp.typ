#import "template.typ": *

#let ih = json("data/induction_heads.json")
#let set-rec(name) = ih.head_sets.find(s => s.set == name).recovered
#let comp(name) = ih.component_decomposition.find(c => c.component == name)

#show: writeup.with(
  section: "§2",
  title: "Mechanistic Interpretability — Reproductions",
  summary: [
    *2.1 Induction heads.* On repeated random tokens, GPT-2 small's loss falls from
    #fx(ih.loss_first_copy, d: 1) to #fx(ih.loss_second_copy) nats on the second copy. The five
    heads with the highest induction score are exactly the five reported in the literature.
    Patching those five heads together into a corrupted run recovers
    #pct(set-rec("top-5 induction heads")) of the loss gap; no single head recovers more than
    #pct(ih.top_patching.at(0).recovered). A decomposition of their inputs separates *where*
    they look from *what* they copy. Restoring only the previous-token heads puts the attention
    back but makes the prediction worse: the heads copy corrupted tokens. The token identity
    they copy comes mostly through MLP 0.
    *2.2 Superposition.* Toy ReLU models reproduce the phase change from "represent the most
    important features" to "pack features into geometric superposition" as sparsity rises.
  ],
)

= Induction heads in GPT-2 small (card 2.1)

Reproduces Olsson et al. (2022), _In-context learning and induction heads_.
Code: `experiments/01_mech_interp/01_induction_heads.py` (about 3 min on CPU).

== Task and metrics

Draw $L = #ih.seq_len$ random tokens $a_1, dots, a_L$ and feed
$t = ("BOS", a_1, dots, a_L, a_1, dots, a_L)$, a batch of #ih.batch sequences. On the first copy
nothing is predictable. On the second copy, every token is predictable by the *induction rule*:
at position $q$, find where the current token $t_q$ occurred before, and predict the token that
followed it. With $A^h in RR^(T times T)$ the attention pattern of head $h$, define
$
  "ind"(h) = 1/L sum_(q=L+1)^(2L) A^h_(q, thin q-L+1), quad
  "prev"(h) = 1/(2L) sum_(q=1)^(2L) A^h_(q, thin q-1).
$
$"ind"(h)$ is the attention a head pays to the token *after* the earlier occurrence, which is
the induction target. $"prev"(h)$ is the attention a head pays to the immediately preceding
token.

For causal evidence we use *activation patching*. A *corrupted* input $t'$ replaces the first
copy with fresh random tokens $b_1, dots, b_L$, so the second copy is no longer a repeat. The
metric is the mean log-probability of the correct next token inside the second copy,
$
  M(t) = 1/(L-1) sum_(q=L+1)^(2L-1) log p_theta (t_(q+1) | t_(<= q)),
$
with $M_"clean" = #fx(ih.clean_logprob)$ and $M_"corr" = #fx(ih.corrupt_logprob)$. Patching
replaces an activation in the corrupted run with its value from the clean run, and we report
the *fraction of the gap recovered*,
$
  r = (M_"patched" - M_"corr") / (M_"clean" - M_"corr"),
$
so $r = 1$ means the patched component carries everything needed and $r = 0$ means nothing.

== Why two heads are needed: composition

An attention head computes $A_(q k) = "softmax"_k (x_q W_Q W_K^top x_k^top \/ sqrt(d_"head"))$
and writes $sum_k A_(q k) x_k W_V W_O$ into the residual stream. The induction mechanism is
two heads in series:
+ A *previous-token head* $h'$ at position $k$ attends to $k - 1$ and writes
  $e(t_(k-1)) W_V^(h') W_O^(h')$ into $x_k$: "the token before me was $t_(k-1)$".
+ An *induction head* $h$ at query $q$ has a QK circuit that matches its own token $e(t_q)$
  against that written information, so the score is high exactly when $t_(k-1) = t_q$. This
  is *K-composition*: the key side reads another head's output. Its OV circuit then copies
  the token $t_k$ found at that key into the logits.

Both halves have to come from the first copy: the *where* (previous-token information in
the keys) and the *what* (the identity of $t_k$ in the values). The last experiment below
separates them.

== Results

#figure(
  image("figures/induction_heads.png", width: 100%),
  caption: [Top left: per-token loss (the dashed line marks the start of the second copy).
    Top right: induction score. Bottom left: previous-token score. Bottom right: fraction of
    the gap recovered by patching each head's output ($z$) alone at all positions.],
) <fig-ih>

*Behaviour.* The loss is #fx(ih.loss_first_copy) nats on the first copy (random tokens) and
#fx(ih.loss_second_copy) nats on the second, so the model is near-perfect at in-context
copying (@fig-ih, top left).

*Attention.* The top five induction scores belong to heads
#ih.top_induction.slice(0, 5).map(h => [#h.head (#fx(h.score))]).join(", "), which are exactly the
induction heads previously reported for GPT-2 small. The next-best, #ih.top_induction.at(5).head,
scores only #fx(ih.top_induction.at(5).score). Head #ih.top_prev_token.at(0).head is an almost
pure previous-token head ($"prev" = #fx(ih.top_prev_token.at(0).score)$), with weaker ones at
#ih.top_prev_token.slice(1).map(h => h.head).join(", ").

*Patching single heads vs head sets.*

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, left, right),
    table.header([patched], [heads], [$r$]),
    ..ih.top_patching.slice(0, 3).map(h => ([single head], [#h.head], [#sfx(h.recovered, d: 3)])).flatten(),
    ..ih.most_negative_patching.slice(0, 2).map(h => ([single head (most negative)], [#h.head], [#sfx(h.recovered, d: 3)])).flatten(),
    ..ih.head_sets.map(s => ([#s.set], [#s.heads], [#sfx(s.recovered, d: 3)])).flatten(),
  ),
  caption: [Fraction of the clean–corrupted gap recovered by patching head outputs.],
) <tab-patch>

#finding[
  The mechanism is *redundant*. Each induction head alone recovers only 10–14%, but the five
  together recover #pct(set-rec("top-5 induction heads")). Single-head patching understates
  circuits spread over parallel heads. Two late heads have *negative* effects: 10.7 and
  11.10. Head 10.7 is the copy-suppression head of McDougall et al. (2023), which down-weights
  tokens that earlier heads are pushing. Patching its clean behaviour in suppresses the right
  answer.
]

== Where vs what: separating K-composition from copying

The previous-token head 4.11 barely registers in @tab-patch
($r = #sfx(set-rec("previous-token head"), d: 3)$) despite a previous-token score of
#fx(ih.top_prev_token.at(0).score). To see why, we patch components at *first-copy positions
only* and measure two quantities: the induction heads' attention to their target (the *where*,
$#fx(ih.corrupted_induction_attention)$ in the corrupted run), and $r$ (the prediction).

#figure(
  image("figures/induction_where_vs_what.png", width: 100%),
  caption: [Left: patching the $k$ strongest previous-token heads in layers 0–4. Attention to
    the induction target climbs steadily while $r$ falls below zero. Right: patching whole
    components at first-copy positions in layers 0–4.],
) <fig-www>

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.header([patched at first-copy positions], [induction attention], [$r$]),
    ..ih.prev_token_sweep.map(s => ([top-#s.k previous-token heads], [#fx(s.induction_attention)], [#sfx(s.recovered, d: 3)])).flatten(),
    ..ih.component_decomposition.map(c => ([#c.component], [#fx(c.induction_attention)], [#sfx(c.recovered, d: 3)])).flatten(),
  ),
  caption: [Where vs what. Clean-run induction attention is #fx(ih.top_induction.slice(0, 5).map(h => h.score).sum() / 5).],
) <tab-www>

#finding[
  - *Previous-token information is spread over many heads.* Head 4.11 alone lifts induction
    attention from #fx(ih.corrupted_induction_attention) to
    #fx(ih.prev_token_sweep.at(0).induction_attention). The top 10 reach
    #fx(ih.prev_token_sweep.at(3).induction_attention), and all 60 heads in layers 0–4 reach
    #fx(ih.prev_token_sweep.at(5).induction_attention).
  - *Restoring the where without the what hurts.* With all 60 heads patched, the induction heads
    attend correctly but copy the *corrupted* tokens $b_k$, and $r$ drops to
    #sfx(ih.prev_token_sweep.at(5).recovered). This isolates the OV half of the circuit.
  - *The what comes through MLP 0.* Patching MLP 0 alone gives $r = #fx(comp("MLP 0 output").recovered)$,
    and restores attention too (#fx(comp("MLP 0 output").induction_attention)). This fits the
    known role of GPT-2's first MLP as an extended token embedding: downstream heads read the
    token identity from it. Attention plus MLP 0 gives #fx(comp("attention + MLP 0").recovered).
]

#caveat[
  One seed, $L = #ih.seq_len$, #ih.batch sequences, and random tokens drawn from ids
  1000–20000. Random-token sequences are off-distribution for GPT-2, so the relative roles of
  the heads could shift on natural text. The next step is to repeat the analysis on repeated
  natural sentences and with resample-ablation.
]

= Toy models of superposition (card 2.2)

#include "parts/02_superposition.typ"

= Your own sparse autoencoder (card 2.3)

#include "parts/02_sae.typ"

= Remaining cards

#status-table(
  (true, [2.1 Induction heads: attention scores, per-head and head-set patching, where-vs-what decomposition.]),
  (true, [2.2 Toy models of superposition: geometry, phase diagram, feature dimensionality.]),
  ("partial", [2.3 SAE trained and compared with the pretrained one. Annotating 5 features by hand is yours (`notes/sae_features_trained.md`).]),
  (false, [2.4 _Original_: hand-trace one single-token completion circuit with activation/path patching. This is your card; the tooling in 2.1 is the starting point.]),
)

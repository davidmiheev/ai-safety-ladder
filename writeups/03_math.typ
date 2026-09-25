#import "template.typ": *

#let MA = json("data/modular_addition.json")
#let MG = json("data/modal_agents.json")
#let H = MA.history
#let first-below(xs, thr) = {
  let i = xs.position(v => v < thr)
  if i == none { none } else { H.step.at(i) }
}
#let grok-step = first-below(H.test_loss, 0.1)
#let memo-step = first-below(H.train_loss, 0.1)
#let restricted-step = first-below(H.restricted_loss, 0.1)
#let share-step = {
  let i = H.top_freq_share.position(v => v > 0.9)
  if i == none { none } else { H.step.at(i) }
}

#show: writeup.with(
  section: "§3",
  title: "Math for AI Safety — Worked Problems",
  summary: [
    *3.1 Modular addition.* A 1-layer transformer trained on 30% of $(a + b) mod #MA.p$
    memorises first (train loss < 0.1 by step #memo-step) and generalises much later (test loss
    < 0.1 by step #grok-step, final test accuracy #pct(MA.final_test_accuracy, d: 1)). The learned
    embedding puts #pct(H.top_freq_share.at(-1)) of its norm on #MA.key_freqs.len() Fourier
    frequencies and lays the residues out on circles. The structure forms before the test loss
    moves. Unlike in the paper, the *output* needs about 20 frequencies, not 5.
    *3.2 Open-source game theory.* An exact evaluator for provability-based agents (modal logic GL on
    Kripke chains) reproduces all #MG.known_checks.len() known outcomes, and gives
    DUPOC vs CUPOD = (#MG.dupoc_vs_cupod.outcome.at(0), #MG.dupoc_vs_cupod.outcome.at(1)) for
    *unbounded* proofs. The open problem is the bounded case, which this does not settle.
  ],
)

= Modular addition and the circular embedding (card 3.1)

Reproduces Nanda et al. (2023), _Progress measures for grokking via mechanistic
interpretability_, in miniature. Code: `experiments/02_math/01_modular_addition.py` (CPU, about
40 min for #H.step.at(-1) steps).

== Setup

Inputs are the three tokens "$a$ $b$ $=$" with $a, b in ZZ_p$ and $p = #MA.p$. The model predicts
$(a + b) mod p$ at "$=$". It is a 1-layer transformer (4 heads, $d_"model" = 128$,
$d_"mlp" = 512$, ReLU, no LayerNorm, no biases), trained full-batch with AdamW (learning rate
$10^(-3)$, weight decay 1) on a random #pct(MA.frac) of the $p^2$ pairs.

== The algorithm the network is expected to find

With $omega_k = 2 pi k \/ p$, the network can solve the task using only trigonometric identities.
Embed each input on a circle, $a |-> (cos omega a, sin omega a)$, for a few *key frequencies*
$k$. The MLP forms products, which by the angle-addition formulas give
$
  cos omega(a + b) = cos omega a cos omega b - sin omega a sin omega b, quad
  sin omega(a + b) = sin omega a cos omega b + cos omega a sin omega b .
$
The unembedding then scores each candidate answer $c$ by
$
  "logit"(c) prop sum_(k in K) cos omega_k (a + b - c)
  = sum_(k in K) [cos omega_k (a+b) cos omega_k c + sin omega_k (a+b) sin omega_k c],
$
which is maximised exactly at $c equiv a + b space (mod p)$. Each frequency alone would also
favour $c$ values near $a + b$ on its own circle; summing several frequencies makes the true
answer the unique peak.

To test this, expand the embedding $W_E in RR^(p times d)$ in the orthonormal real Fourier basis
of $ZZ_p$ and measure the share of $norm(W_E)_F^2$ carried by each frequency $k$ (the cos and sin
components together). The *restricted loss* projects the logits onto
$cos omega_k (a + b - c)$ and $sin omega_k (a + b - c)$ for the top-#MA.key_freqs.len() frequencies
only, and evaluates the test loss of that projection. It is low only if those few terms already
implement the algorithm.

== Results

#figure(
  image("figures/modular_addition.png", width: 100%),
  caption: [Top left: train, test and restricted test loss (log scale). Top right: share of
    the embedding norm in the top-#MA.key_freqs.len() frequencies, and the total weight norm.
    Bottom left: Fourier spectrum of the final embedding (key frequencies highlighted). Bottom right:
    the embedding projected on the plane of frequency #MA.circle_freq, coloured by
    $#MA.circle_freq dot a mod p$.],
) <fig-modadd>

#let OS = MA.output_side
#let MO = json("data/modular_addition_output.json")
#let sw(k) = MO.sweep.find(r => r.k == k)
#finding[
  - *Grokking.* Train loss falls below 0.1 by step #memo-step\; test loss only by step #grok-step.
    Final test accuracy is #pct(MA.final_test_accuracy, d: 2).
  - *A sparse Fourier embedding.* The embedding puts #pct(H.top_freq_share.at(-1)) of its norm on
    frequencies #MA.key_freqs.map(str).join(", "). A random embedding would spread its norm
    evenly, about #pct(MA.key_freqs.len() / ((MA.p - 1) / 2)) on any #MA.key_freqs.len() frequencies.
  - *Circles.* On the plane of frequency #MA.circle_freq, the #MA.p residues lie on a circle
    #if MA.circle_freq == 1 [in their natural order $a$ (for other key frequencies $k$ the order
    is $k a mod p$)] else [in the order $#MA.circle_freq a mod p$, not in the order $a$]. This
    is the "feature manifold" of the reading list's Interpretability & Feature Manifolds direction.
  - *Neurons compute functions of $a + b$.* Median $R^2 = #fx(MA.neuron_freq_r2.median_r2)$, and
    #pct(MA.neuron_freq_r2.at("frac_above_0.9")) of the #MA.neuron_freq_r2.n_neurons neurons exceed 0.9.
  - *The structure appears before generalisation.* The embedding's Fourier share rises steadily
    (@fig-modadd, top right) while test loss is still above 10 nats, and passes 0.9 at step
    #if share-step == none [—] else [#share-step].
]

== The output is not five clean cosines

Nanda et al.'s second progress measure, the *restricted loss*, keeps only the key frequencies of
the logits. Here it does not behave as a progress measure: it dips at grokking and then rises to
about 3 nats while the true test loss stays near $10^(-3)$ (@fig-modadd, dashed). I checked the
implementation against an independent FFT version, and the two agree exactly, so this is a
property of the model. Diagnostics on the final logits, taken as a function of the answer $c$
(`experiments/02_math/01b_output_frequencies.py` for the sweep over $K$):

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    table.header([logits over $c$, test set], [value]),
    [full model: test loss], [#fx(OS.test_loss, d: 4)],
    [top frequencies of the logits (by power)], [#OS.logit_top_freqs.map(str).join(", ")],
    [power share of those 5], [#pct(OS.logit_share_in_logit_top)],
    [power share of the embedding's 5 key frequencies], [#pct(OS.logit_share_in_embedding_keys)],
    [loss keeping only the logits' top 5], [#fx(OS.restricted_loss_logit_keys, d: 2)],
    [loss keeping only the embedding's 5 keys], [#fx(OS.restricted_loss_embedding_keys, d: 2)],
    [loss *removing* the logits' top 5], [#fx(OS.excluded_loss_logit_keys, d: 1)],
  ),
  caption: [Where the logits' information lives. Uniform guessing is $ln #MA.p approx #fx(calc.ln(MA.p))$ nats.],
) <tab-output>

#finding[
  - *Power sits in low frequencies, which barely discriminate.* Frequency 1 alone carries
    #pct(sw(1).power_share) of the logit power, and $k = 1, 2, 3$ together carry
    #pct(MO.low_freq_power_share). A cosine of frequency 1 changes by only
    $1 - cos(2 pi \/ #MA.p) approx #fx(1 - calc.cos(2 * calc.pi / MA.p), d: 4)$ of its amplitude
    between neighbouring answers.
  - *About 20 frequencies are needed.* Keeping the top 5 by power (#pct(sw(5).power_share) of it)
    leaves the model at chance: loss #fx(sw(5).loss, d: 1), argmax right #pct(sw(5).accuracy) of
    the time. The top 10 give #pct(sw(10).accuracy) accuracy, and the top 20 give
    #pct(sw(20).accuracy) (loss #fx(sw(20).loss)).
  - *Weak evidence for combination frequencies.* #MO.hits.len() of the top 20 logit frequencies
    are embedding keys or their pairwise sums and differences (#MO.hits.map(str).join(", ")),
    against #fx(MO.null_mean_hits, d: 1) expected by chance ($p = #fx(MO.p_value, d: 3)$, 5,000
    random draws). That is consistent with the MLP's ReLUs creating product terms that the
    unembedding reads, on top of the clean $cos omega(a+b-c)$ terms, but it's one model and a
    borderline test.
]

#caveat[
  - Nanda et al. trained for 40k steps; this run stopped at #H.step.at(-1). Weight decay keeps
    pruning after grokking, so the cleaner five-frequency output may simply need more training.
    That's cheap to test with `--steps 50000`.
  - Two runs with the same seed found different key frequencies (1, 49, 21, 7, 14 here; 49, 3, 1,
    21, 42 in an earlier run). Multi-threaded CPU arithmetic isn't bit-reproducible, and grokking
    amplifies tiny differences. Frequencies 1, 49 and 21 appeared in both runs. The circle,
    grokking and neuron findings held in both.
]

= Open-source game theory with provability (card 3.2)

Code: `experiments/02_math/02_modal_agents.py`. Source: mathforaisafety.org, _Open-Source Game
Theory_, and LaVictoire et al. (2014), _Robust Cooperation in the Prisoner's Dilemma_.

== Agents as modal formulas

Two programs play a one-shot Prisoner's Dilemma, each able to read the other's source. An agent
that decides by searching for *proofs* about the matchup is a formula of provability logic,
where $square phi$ reads "PA proves $phi$". Writing $[X(Y)]$ for "$X$ cooperates with $Y$":
$
  [#smallcaps[FairBot]\(Y)] <-> square [Y(#smallcaps[FairBot])], quad
  [#smallcaps[Cupod]\(Y)] <-> not square not [Y(#smallcaps[Cupod])],
$
$
  [#smallcaps[PrudentBot]\(Y)] <-> square [Y(#smallcaps[PrudentBot])] and
  square (not square bot -> not [Y(#smallcaps[DefectBot])]).
$
FairBot is DUPOC ("defect unless proof of cooperation"). CUPOD is "cooperate unless proof of
defection".

Every agent variable appears under a $square$, so the system of equivalences has a unique
solution (de Jongh–Sambin fixed-point theorem). Because GL is complete for PA's provability
(Solovay), that solution can be computed exactly. Evaluate on the linear Kripke frame
$0 < 1 < 2 < dots$, where $square phi$ holds at world $n$ iff $phi$ holds at every world $m < n$.
World $n$ then depends only on earlier worlds, the values become constant after finitely many
worlds, and the constant value is the actual outcome. At world 0 every $square phi$ is
vacuously true. That is the Löbian "benefit of the doubt" that lets FairBot cooperate with itself.

== Results

#let names = MG.agents
#figure(
  table(
    columns: (auto,) + (1fr,) * names.len(),
    align: center,
    table.header([row vs column], ..names.map(n => text(size: 8pt, n))),
    ..names.map(a => ([#text(size: 8pt, a)], ..names.map(b => [#MG.table.at(a + "|" + b)]))).flatten(),
  ),
  caption: [The row agent's move against the column agent (unbounded proofs). Computed, not assumed.],
) <tab-modal>

#finding[
  - *All #MG.known_checks.len() known outcomes reproduce*, including the subtle ones.
    PrudentBot exploits CooperateBot, while FairBot does not. CUPOD defects against *itself*, even
    though it seems the "nicer" bot: $p <-> not square not p$ has the unique solution $p <-> bot$.
  - *DUPOC vs CUPOD = (#MG.dupoc_vs_cupod.outcome.at(0), #MG.dupoc_vs_cupod.outcome.at(1))* in the
    unbounded setting, stable from world #MG.dupoc_vs_cupod.stable_from_world. At world 0, DUPOC
    cooperates (vacuous $square$), and so $not square not$ fails for CUPOD, which defects. From
    world 1 on, $square$[CUPOD cooperates] is false because of world 0, so DUPOC defects. $not square not$[DUPOC
    cooperates] is then true because DUPOC cooperated at world 0, so CUPOD cooperates. This is the
    GL version of the page's argument: PA can't prove that CUPOD cooperates with anyone, so
    DUPOC defects, and CUPOD, finding no proof of defection, cooperates.
]

#caveat[
  - *This does not settle the open problem.* The problem is about *bounded* proofs, $square_k$
    ("a proof of length $<= k$"), where GL's axioms fail. The known bounded results (DUPOC–DUPOC
    cooperate, CUPOD–CUPOD defect) need the parametric bounded Löb theorem, and the obstacle cited
    for DUPOC(k) vs CUPOD(k) is the fixed point $X <-> not square_k not square_k X$. The GL answer
    is evidence for "(D,C) eventually", not a proof.
  - *A likely erratum on the source page.* It says "both CUPOD and DUPOC cooperate against
    FairBot" but also defines FairBot as DUPOC, and derives that DUPOC *defects* against CUPOD.
    Those claims can't all hold. In the table, CUPOD cooperates with FairBot but FairBot
    defects against CUPOD.
]

*Next (your original work):* a toy *bounded* proof search, for example a small proof system
with explicit proof-length bounds where □ₖ is decided by search, checked first against the
known small-$k$ outcomes above, and then run on DUPOC(k) vs CUPOD(k) as $k$ grows.

= No-coincidence argument (card 3.3)

Original card. The statement of the conjecture, the exercise and a starting point are in
`experiments/02_math/03_no_coincidence.md`. The network from 3.1 is a natural first candidate:
it is correct on all $p^2$ inputs, which for random weights would be a coincidence of probability
about $p^(-p^2)$, and the Fourier circuit above is its structural explanation.

= Status of §3

#status-table(
  (true, [3.1 Modular addition: grokking, Fourier embedding, circles, neurons, and an output-side analysis of the restricted-loss discrepancy.]),
  ("partial", [3.2 Modal agents: exact unbounded evaluator and known results done. Bounded proof search is yours.]),
  (false, [3.3 No-coincidence mini-argument: scaffold in `03_no_coincidence.md`.]),
)

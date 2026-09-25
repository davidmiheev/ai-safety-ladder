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
    frequencies and lays the residues out on circles. The Fourier structure is measurable long
    before the test loss moves.
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

#finding[
  - *Grokking.* Train loss falls below 0.1 by step #memo-step; test loss only by step #grok-step.
  - *Sparse Fourier structure.* The final embedding puts #pct(H.top_freq_share.at(-1)) of its norm on
    frequencies #MA.key_freqs.map(str).join(", "). A random embedding spreads its norm evenly,
    about #pct(MA.key_freqs.len() / ((MA.p - 1) / 2)) on any #MA.key_freqs.len() frequencies.
  - *Circles.* On the plane of frequency #MA.circle_freq, the #MA.p residues lie on a circle in
    the order $#MA.circle_freq a mod p$, not in the order $a$. This is the "feature manifold"
    of the reading list's Interpretability & Feature Manifolds direction.
  - *Neurons compute functions of $a + b$.* For each MLP neuron at "$=$", we fit its activation
    over all $p^2$ inputs with cos/sin of $omega_k a$, $omega_k b$ and $omega_k (a+b)$ at its best
    key frequency. Median $R^2 = #fx(MA.neuron_freq_r2.median_r2)$, and
    #pct(MA.neuron_freq_r2.at("frac_above_0.9")) of the #MA.neuron_freq_r2.n_neurons neurons
    exceed 0.9.
  - *Progress measure.* The restricted test loss falls below 0.1 at step
    #if restricted-step == none [—] else [#restricted-step], and the Fourier share passes 0.9 at step
    #if share-step == none [—] else [#share-step]. Test loss follows at step #grok-step.
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
  (true, [3.1 Modular addition: grokking, Fourier spectrum, circles, progress measures.]),
  ("partial", [3.2 Modal agents: exact unbounded evaluator and known results done. Bounded proof search is yours.]),
  (false, [3.3 No-coincidence mini-argument: scaffold in `03_no_coincidence.md`.]),
)

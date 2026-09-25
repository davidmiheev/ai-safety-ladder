#import "../template.typ": *
#let TA = json("../data/toy_superposition_a.json")
#let A = TA.rows
#let B = json("../data/toy_superposition_b.json")
#let C = json("../data/toy_superposition_c.json")

Reproduces Elhage et al. (2022), _Toy Models of Superposition_.
Code: `experiments/01_mech_interp/02_toy_superposition.py`. All parts ran on a Colab T4 through
the MCP server: 17 s for (a), 47 s for (b), and 5 min for (c). The figures were redrawn locally from
the saved JSON with `--replot`.

== Model

$n$ features are compressed into $m < n$ hidden dimensions and read back out with a tied
weight matrix $W in RR^(m times n)$:
$
  h = W x, quad x' = "ReLU"(W^top h + b), quad
  cal(L) = EE_x sum_(i=1)^n I_i (x_i - x'_i)^2 .
$
Each feature is present independently with probability $p = 1 - S$ (the *density*; $S$ is the
*sparsity*), and then $x_i tilde U[0, 1]$. $I_i$ is its *importance*. Expanding coordinate $i$
shows where the tension comes from:
$
  x'_i = "ReLU"( norm(W_i)^2 x_i + sum_(j != i) (W_i dot W_j) x_j + b_i ),
$ <eq-interf>
where $W_i$ is column $i$. With $m < n$ the columns cannot all be orthogonal, so representing
more than $m$ features costs *interference* $(W_i dot W_j) x_j$. When features are sparse, two
rarely occur together, so the interference term is usually zero. The ReLU with a negative bias
removes the small positive interference that remains. So sparsity should make superposition
worth its cost. All experiments train many small models in parallel (Adam, learning rate
$10^(-3)$ with cosine decay, batch 1024) and keep the best of several seeds.

== (a) Geometry: 5 features in 2 dimensions

With $n = 5$, $m = 2$ and $I_i = 0.9^i$:

#figure(
  image("../figures/superposition_geometry.png", width: 100%),
  caption: [Top: columns $W_i in RR^2$ (dark = most important). Bottom: $W^top W$ (blue
    positive, red negative). Features "represented" means $norm(W_i) > 0.5$.],
) <fig-geom>

#figure(
  table(
    columns: (auto,) * (A.len() + 1),
    align: center,
    table.header([$S$], ..A.map(r => [#r.sparsity])),
    [features represented], ..A.map(r => [#r.n_represented]),
    [loss], ..A.map(r => [#fx(r.loss, d: 3)]),
  ),
  caption: [Represented features and final loss for the best of #TA.seeds seeds per sparsity.],
)

#finding[
  Three regimes, as in the paper:
  - *Dense* ($S = 0$): the textbook PCA-like solution. The #A.at(0).n_represented most important
    features take orthogonal directions and the rest are dropped.
  - *Intermediate* ($S = 0.7$–$0.97$): four features form two *antipodal pairs*, an "X" whose
    $W^top W$ has $-1$ entries.
  - *Sparse* ($S >= 0.99$): all five sit on a regular *pentagon*, with norms rising to about
    #fx(A.at(-1).norms.at(0)) and a negative bias of about #fx(A.at(-1).bias.at(0)). The bias
    is the ReLU threshold that clips the positive interference in @eq-interf.

  Represented features per sparsity: #A.map(r => str(r.n_represented)).join(" → ").
]

#caveat[
  The dense and sparse ends were identical across three independent runs. The count in the
  transition zone ($S = 0.7$–$0.97$) was not: it ranged from 3 to 5 depending on the seeds,
  because several configurations reach nearly equal loss there. The feature dropped in the
  intermediate regime is \#3 rather than the least important \#4. Their importances
  ($0.9^3$ and $0.9^4$) differ by only 10%, so which one loses is close to a coin flip.
]

== (b) Phase diagram: 2 features in 1 dimension

With $n = 2$ and $m = 1$, $W = (w_1, w_2)$ is just two numbers. Feature 1 has importance 1 and
feature 2 has importance $I_2$. We sweep $I_2 in [0.1, 10]$ and the density $1 - S in [0.01, 1]$
on a #B.grid × #B.grid log grid, with #B.seeds seeds per cell, and classify the best model by
which $|w_i| > 0.5$.

#figure(
  image("../figures/superposition_phase.png", width: 62%),
  caption: [Which features the 1-dimensional model represents, as a function of density and
    relative importance.],
) <fig-phase>

#finding[
  Three phases, as in the paper (cells: only feature 1: #B.counts.only_f1, only feature 2:
  #B.counts.only_f2, both: #B.counts.superposition):
  - With *dense* features the model is a hard switch at $I_2 \/ I_1 = 1$: it keeps whichever
    feature is more important.
  - As density falls, superposition takes over. The more unequal the importances, the sparser
    the features must be first: from about $1 - S approx 0.5$ at equal importance to about
    $0.1$–$0.2$ at a 10× ratio. At very low density the model stores both features even when
    one matters 10× less, because they almost never co-occur.
  - *Every* superposed solution is antipodal (#pct(B.antipodal_fraction), $w_1 w_2 < 0$). With
    opposite signs, the interference when one feature is active is negative, and the ReLU clips
    it to zero at no cost.
]

#caveat[
  The isolated off-phase cells deep inside the superposition region are optimisation noise, not
  phases. There, every loss scales with the density, so the gap between configurations is tiny
  (about $10^(-3)$) relative to the minibatch noise of #TA.seeds seeds × $10^4$ steps.
]

== (c) How many dimensions does a feature get?

For $n = #C.n$ features in $m = #C.m$ dimensions with uniform importance, two measures:
$
  D^* = m / norm(W)_F^2 quad "(dimensions per represented feature)", quad
  D_i = norm(W_i)^2 / (sum_j (hat(W)_i dot W_j)^2),
$
where $hat(W)_i = W_i / norm(W_i)$. $D_i$ is the fraction of a dimension that feature $i$ has to
itself: $1$ for a dedicated direction, $1/2$ for each member of an antipodal pair, $2/3$ for a
triangle, $2/5$ for a pentagon, $3/4$ for a tetrahedron and $3/8$ for a square antiprism.

#figure(
  image("../figures/superposition_dimensionality.png", width: 100%),
  caption: [Left: $D^*$ against $1/(1-S)$; dotted lines mark the configuration fractions.
    Right: $D_i$ for every represented feature at each sparsity.],
) <fig-dim>

#let plat = range(C.density.len()).filter(j => calc.abs(C.dims_per_feature.at(j) - 0.5) < 0.02)
#let plat-feats = plat.map(j => C.D_represented.at(j)).flatten()
#let plat-exact = plat-feats.filter(d => calc.abs(d - 0.5) < 0.01).len()
#let last-D = C.D_represented.at(-1)
#let last-mean = last-D.sum() / last-D.len()
#let early = range(C.density.len()).filter(j => 1 / C.density.at(j) > 1.1 and 1 / C.density.at(j) < 1.7)
#let early-feats = early.map(j => C.D_represented.at(j)).flatten()
#let near(x, v) = calc.abs(x - v) < 0.03

#finding[
  The curve is not smooth. It has the regimes the paper describes:
  - *Dense* ($S = 0$): $D^* = #fx(C.dims_per_feature.at(0))$. The model uses all $m$ dimensions,
    but with uniform importance it learns a rank-$m$ projection spread thinly across many
    features, as PCA would. Only #C.n_represented.at(0) features reach norm 0.5.
  - *Low sparsity* ($1 \/ (1-S) approx 1.2$–$1.6$): of the #early-feats.len() represented
    features, #early-feats.filter(d => near(d, 3 / 4)).len() sit at $D_i approx 3/4$
    (tetrahedra) and #early-feats.filter(d => near(d, 2 / 3)).len() at $approx 2/3$ (triangles).
  - *A plateau at exactly $1/2$* for $1 \/ (1-S) in [#fx(1 / C.density.at(plat.first())),
    #fx(1 / C.density.at(plat.last()))]$: in those #plat.len() models, #plat-exact of
    #plat-feats.len() represented features have $|D_i - 1/2| < 0.01$. Every feature is in an
    antipodal pair.
  - *Sparse*: after brief clusters near $2/5$ and $3/8$, $D_i$ decreases continuously until
    #C.n_represented.at(-1) of the #C.n features are represented, with
    $D_i = #fx(last-mean, d: 3) = m \/ n$.

  Over the whole sweep only #pct(C.fraction_near_sticky) of feature–sparsity points lie within
  $0.03$ of a configuration value
  (#C.near_sticky_counts.pairs().filter(((k, v)) => v > 0).map(((k, v)) => [#k: #v]).join(", ")),
  because the sparse end is a continuum. Where the model is still *choosing* a geometry, it
  chooses discrete ones.
]

#caveat[
  Part (c) is one seed per sparsity at $#C.steps$ steps and $n = #C.n$, versus the paper's larger
  models. The sticky points beyond $1/2$ (such as $2/5$ and $3/8$) are therefore only transient
  here. A longer run with $n = 400$ would show whether they widen into plateaus. The earlier
  $10^4$-step run found the same $1/2$ plateau but blurrier edges.
]

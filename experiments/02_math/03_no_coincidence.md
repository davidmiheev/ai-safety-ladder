# 3.3 (original) — a no-coincidence argument in miniature

Source: https://mathforaisafety.org/research/heuristic-estimators (ARC's no-coincidence conjecture).

## The conjecture, as stated there

> Let C : {0,1}³ⁿ → {0,1}³ⁿ be a reversible circuit. Define P(C) to mean: there is no input x
> ending in n zeros such that C(x) also ends in n zeros.
>
> There is a polynomial-time verifier program V which takes in a pair (C, π) as inputs, such that:
> **Completeness:** if P(C) holds, then some polynomial-length advice string π makes V(C, π) = 1.
> **Random-circuit soundness:** for 99% of random reversible circuits C, no advice string π makes V(C, π) = 1.

With 100% instead of 99% it would be equivalent to NP = coNP. The "99%" is the heuristic part:
the explanation π only has to rule out *coincidence*, not prove P(C) for every circuit.

## The exercise

You're not proving the conjecture. Build the kind of concrete example it formalises:

1. **A property that would be a wild coincidence for a random network.** It holds on
   exponentially many (or all) inputs, and a random network of the same architecture almost
   surely violates it.
2. **A structural explanation π**: a short description of *why* it holds, much shorter than
   checking every input.
3. **A verifier V**: code that takes (network, π) and checks π's claims in time polynomial in
   the network size, *without* enumerating the inputs.
4. **Soundness in miniature**: run V on random networks of the same architecture with the same
   (or a best-effort adapted) π and show that it rejects them. Report the rejection rate.
5. **An estimate that beats sampling** (the Matching Sampling Principle): use π to estimate some
   E_x[f(M(x))] and compare its error with naive sampling at equal compute.

## A starting point from this repo (not the answer)

The modular-addition network from card 3.1 is correct on all p² inputs. For random weights that's
a coincidence with probability ≈ (1/p)^(p²). Its explanation is the trigonometric circuit: a few
key frequencies, cos/sin embeddings, product-to-sum in the MLP. Questions to settle:
what exactly π contains, what V can check about the weights in polynomial time, and where the
argument has to become heuristic (the MLP's ReLUs only *approximately* compute the products).

Or pick your own: a sorting network, a parity circuit with a planted structure, an induction
head (§2.1) as "copies correctly on every repeated sequence".

Write-up: add a section to `writeups/03_math.typ`.

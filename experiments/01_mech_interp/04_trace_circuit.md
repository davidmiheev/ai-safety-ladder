# 2.4 (original) — hand-trace a circuit

No paper to check against by design. The tooling (`src/ladder/patching.py`,
`04_trace_circuit.py`) is validated on IOI: run `tasks/ioi_example.json` and you should recover
the name movers (9.9, 9.6, 10.0), S-inhibition (7.9, 8.6, 8.10), induction heads feeding
duplicate-token detection (5.5, 6.9), and the negative name movers (10.7, 11.10). Then **pick
something else**.

## 1. Pick a behaviour

A single-token completion that GPT-2 small gets right with a clear margin (logit diff > 2),
where you can build a corrupted prompt that changes *only* the information the behaviour uses.
Ideas, roughly easiest first:

- **Closing brackets / quotes**: `"He said (quietly" → ")"` vs a prompt with no open bracket.
- **Plural–verb agreement**: `The keys to the cabinet → are / is`, corrupt by making the subject singular.
- **Greater-than** (Hanna et al., 2023, if you want a known answer to compare with afterwards):
  `The war lasted from 1732 to 17 → YY > 32`.
- **Acronyms**: `The Federal Bureau of Investigation (F → BI`.
- **Gendered pronouns**: `Mary went home because → she / he`.

## 2. Method

1. Write 8–20 pairs in a task JSON; same token length within clean and within corrupt.
   Check `clean logit diff` ≫ 0 and `corrupt logit diff` ≪ 0 (or at least well apart).
2. `resid_pre` map: which positions carry the information, and from which layer on is it at the
   final position? The layer where it jumps to the final position is where attention moves it.
3. Head map: the heads with large |effect|. Negative heads are real findings, not noise.
4. For each important late head, look at its attention pattern (CircuitsVis, as in §1): which
   token does it attend to from the final position?
5. Path patching, working backwards from the logits:
   `--senders <late heads> --receivers logits`, then
   `--senders <earlier heads> --receivers <late head>q` (and `k`, `v`), and so on.
   Each hop should explain a large share of the previous one.
6. State the mechanism in one paragraph: which head moves what information from where to where,
   and which head reads it. Predict something new from it (a prompt it should fail on) and test it.

## 3. Checklist before calling it done

- [ ] Clean vs corrupt differ only in the causal variable (no length or position confounds).
- [ ] Results hold across ≥ 8 prompt pairs, not one.
- [ ] Tried a second corruption (e.g. resample other names or objects), not just one flip.
- [ ] Every head in the story has a path-patching number, not just a head-map number.
- [ ] The mechanism made a prediction that you then tested.
- [ ] Wrote down what you *couldn't* explain (the negative heads, the leftover effect).

Write-up: add it to `writeups/` like the other sections, with the maps from
`outputs/01_mech_interp/circuit_<name>/`.

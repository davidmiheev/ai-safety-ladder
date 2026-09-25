"""3.2 — Open-source Prisoner's Dilemma between provability-based agents (modal agents in GL).

Agents that decide by searching for proofs about each other ("cooperate iff I can prove you
cooperate with me") can be written as *modal formulas*: □φ reads "PA proves φ". Provability
logic GL is sound and complete for PA's provability predicate (Solovay), and a system of
*modalized* fixed-point equations (every agent variable under a □) has a unique solution. That
solution can be computed on finite linear Kripke frames (worlds 0 < 1 < 2 < ...): □φ holds at
world n iff φ holds at every world m < n. Values stabilise after finitely many worlds, and the
stable value is the actual outcome. (LaVictoire, Fallenstein, Yudkowsky, Bárász, Christiano,
Herreshoff 2014, "Robust Cooperation in the Prisoner's Dilemma".)

This is the *unbounded* setting. The open problem on mathforaisafety.org is about *bounded*
proofs (□_k = "a proof of length ≤ k"), where GL no longer applies. This script:
  1. verifies the known unbounded results (FairBot, PrudentBot, DefectBot, CooperateBot);
  2. reproduces DUPOC–DUPOC = (C,C) and CUPOD–CUPOD = (D,D);
  3. computes DUPOC vs CUPOD in GL: the unbounded analogue of the open question.

  uv run python experiments/02_math/02_modal_agents.py
"""

import json
from dataclasses import dataclass
from itertools import product

from ladder import output_dir

# --- Modal formulas ------------------------------------------------------------------------------


@dataclass(frozen=True)
class F:
    op: str            # "var" "top" "bot" "not" "and" "or" "box"
    args: tuple = ()
    name: str = ""

    def __and__(self, o): return F("and", (self, o))
    def __or__(self, o): return F("or", (self, o))
    def __invert__(self): return F("not", (self,))

    def __str__(self):
        if self.op == "var":
            return self.name
        if self.op in ("top", "bot"):
            return "⊤" if self.op == "top" else "⊥"
        if self.op == "not":
            return f"¬{self.args[0]}"
        if self.op == "box":
            return f"□{self.args[0]}"
        return f"({self.args[0]} {'∧' if self.op == 'and' else '∨'} {self.args[1]})"


TOP, BOT = F("top"), F("bot")


def V(name: str) -> F:
    return F("var", name=name)


def box(f: F) -> F:
    return F("box", (f,))


def implies(a: F, b: F) -> F:
    return ~a | b


def modalized(f: F, under_box: bool = False) -> bool:
    if f.op == "var":
        return under_box
    return all(modalized(a, under_box or f.op == "box") for a in f.args)


# --- Agents ------------------------------------------------------------------------------------------
# An agent is a function (me, opp, coop) -> formula for "me cooperates with opp", where coop(x, y)
# returns the variable "x cooperates with y". Agents may refer to other matchups (PrudentBot does).

def cooperate_bot(me, opp, coop): return TOP
def defect_bot(me, opp, coop): return BOT
def fair_bot(me, opp, coop): return box(coop(opp, me))                   # = DUPOC
def cupod(me, opp, coop): return ~box(~coop(opp, me))                      # cooperate unless proof of defection
def prudent_bot(me, opp, coop):
    # C iff it can prove the opponent cooperates with it AND (with one extra level of consistency)
    # that the opponent defects against DefectBot.
    return box(coop(opp, me)) & box(implies(~box(BOT), ~coop(opp, "DefectBot")))


AGENTS = {"CooperateBot": cooperate_bot, "DefectBot": defect_bot, "FairBot": fair_bot,
          "DUPOC": fair_bot, "CUPOD": cupod, "PrudentBot": prudent_bot}


def solve(pairs: list[tuple[str, str]], max_worlds: int = 40):
    """Build the fixed-point system for the given matchups (plus everything they reference) and
    evaluate it on the linear Kripke frame. Returns ({(x, y): bool}, stabilisation world)."""
    eqs: dict[tuple[str, str], F] = {}
    todo = list(pairs)
    while todo:
        x, y = todo.pop()
        if (x, y) in eqs:
            continue
        refs = []

        def coop(a, b, refs=refs):
            refs.append((a, b))
            return V(f"{a}|{b}")

        eqs[(x, y)] = AGENTS[x](x, y, coop)
        if not modalized(eqs[(x, y)]):
            raise ValueError(f"{x} vs {y}: not modalized, fixed point not guaranteed")
        todo += refs

    history: list[dict[str, bool]] = []  # history[n][var] = value at world n

    def ev(f: F, n: int, cur: dict[str, bool]) -> bool:
        if f.op == "top":
            return True
        if f.op == "bot":
            return False
        if f.op == "var":
            return cur[f.name]  # only reached under a box, i.e. via history, see below
        if f.op == "not":
            return not ev(f.args[0], n, cur)
        if f.op == "and":
            return ev(f.args[0], n, cur) and ev(f.args[1], n, cur)
        if f.op == "or":
            return ev(f.args[0], n, cur) or ev(f.args[1], n, cur)
        if f.op == "box":  # true iff the body holds at every earlier world
            return all(ev(f.args[0], m, history[m]) for m in range(n))
        raise ValueError(f.op)

    stable_from = None
    for n in range(max_worlds):
        # Modalized: variables only appear under □, so world n depends only on worlds < n.
        cur = {f"{x}|{y}": ev(f, n, {}) for (x, y), f in eqs.items()}
        history.append(cur)
        if n == 0 or cur != history[n - 1]:
            stable_from = n  # first world of the current constant run
    final = history[-1]
    return {k: final[f"{k[0]}|{k[1]}"] for k in eqs}, stable_from, {f"{k[0]} vs {k[1]}": str(f) for k, f in eqs.items()}


def outcome(a: str, b: str):
    vals, depth, _ = solve([(a, b), (b, a)])
    return ("C" if vals[(a, b)] else "D", "C" if vals[(b, a)] else "D"), depth


# Known results. Unbounded, from LaVictoire et al. 2014; CUPOD/DUPOC from mathforaisafety.org.
KNOWN = {
    ("FairBot", "FairBot"): ("C", "C"),
    ("FairBot", "DefectBot"): ("D", "D"),
    ("FairBot", "CooperateBot"): ("C", "C"),
    ("PrudentBot", "PrudentBot"): ("C", "C"),
    ("PrudentBot", "FairBot"): ("C", "C"),
    ("PrudentBot", "CooperateBot"): ("D", "C"),
    ("PrudentBot", "DefectBot"): ("D", "D"),
    ("DUPOC", "DUPOC"): ("C", "C"),
    ("CUPOD", "CUPOD"): ("D", "D"),
}


def main() -> None:
    names = ["CooperateBot", "DefectBot", "FairBot", "CUPOD", "PrudentBot"]
    table, depths = {}, {}
    for a, b in product(names, repeat=2):
        (oa, _), d = outcome(a, b)
        table[f"{a}|{b}"] = oa
        depths[f"{a}|{b}"] = d

    print("row agent's move against column agent (unbounded, GL):")
    print(" " * 14 + "".join(f"{n:>13}" for n in names))
    for a in names:
        print(f"{a:>14}" + "".join(f"{table[f'{a}|{b}']:>13}" for b in names))

    checks = []
    for (a, b), want in KNOWN.items():
        got, _ = outcome(a, b)
        checks.append({"pair": f"{a} vs {b}", "expected": "".join(want), "got": "".join(got), "ok": got == want})
        print(f"  {'✓' if got == want else '✗'} {a} vs {b}: {got} (known {want})")
    assert all(c["ok"] for c in checks), "a known result failed to reproduce"

    (d_move, c_move), depth = outcome("DUPOC", "CUPOD")
    _, _, eqs = solve([("DUPOC", "CUPOD"), ("CUPOD", "DUPOC")])
    print(f"\nDUPOC vs CUPOD (unbounded): ({d_move},{c_move}), stable from world {depth}")
    for k, f in eqs.items():
        print(f"  {k}: C ↔ {f}")

    out = output_dir("02_math")
    res = {"table": table, "stable_from_world": depths, "agents": names, "known_checks": checks,
           "dupoc_vs_cupod": {"outcome": f"{d_move}{c_move}", "stable_from_world": depth, "equations": eqs}}
    (out / "modal_agents.json").write_text(json.dumps(res, indent=1))
    print(f"-> {out / 'modal_agents.json'}")


if __name__ == "__main__":
    main()

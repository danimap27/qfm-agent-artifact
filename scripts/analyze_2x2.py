"""Evidence x citation-contract 2x2 analysis at the seed level.

Cells: grounded (evidence, contract) and free (neither) come from the frozen
grid, evidence_only and cite_only from results/grid_2x2. Seed is the analysis
unit: per model, the mean over templates and requests gives one value per seed
and condition, and every contrast is an exact Wilcoxon test over the 10 seeds.

Estimands, named by the factor that changes between the two cells:
- UR is conditioned on the contract (cells without it sit at 1.000), so it is
  decomposed sequentially: contract alone (free -> cite_only) and evidence
  given the contract (cite_only -> grounded).
- Claims vary freely, so the four simple effects are reported.
- Prominence exists only where attributions exist, so the one contrast is the
  contract given the evidence (evidence_only -> grounded).
- ur_split rescores with '[a, b]' citations split into '[a] [b]', a sensitivity
  check on the strict one-field-per-bracket rule of the scorer.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.eval.claims import score_explanation
from code.eval.value_audit import check_value_claims
from code.verbalizer.prompts_2x2 import CITE_ONLY_FIELDS

# Optional argument: one root holding all four cells (the classical control).
SINGLE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
GRID, GRID2 = (SINGLE, SINGLE) if SINGLE else (pathlib.Path("results/grid_q1"), pathlib.Path("results/grid_2x2"))
CELLS = {"grounded": GRID, "free": GRID, "evidence_only": GRID2, "cite_only": GRID2}
MULTI = re.compile(r"\[([A-Za-z0-9_.\-]+(?:\s*,\s*[A-Za-z0-9_.\-]+)+)\]")


def split_citations(text: str) -> str:
    return MULTI.sub(lambda m: " ".join(f"[{p.strip()}]" for p in m.group(1).split(",")), text)


assert split_citations("x [a.b, c.d].") == "x [a.b] [c.d]."

rows = defaultdict(list)  # (model, seed, condition) -> per-explanation dicts
for root in dict.fromkeys((GRID, GRID2)):
    for cell in sorted(root.glob("*/*/seed*")):
        exp = cell / "explanations.jsonl"
        if not exp.exists():
            continue
        logs = {json.loads(l)["request_id"]: json.loads(l) for l in open(cell / "logs.jsonl") if l.strip()}
        model, seed = cell.parts[-3], cell.parts[-1]
        for line in open(exp):
            if not line.strip():
                continue
            rec = json.loads(line)
            cond = rec["condition"]
            if CELLS.get(cond) != root:
                continue
            text = rec["parsed"].get("explanation") or rec["response"]
            log = logs[rec["request_id"]]
            kinds = defaultdict(lambda: [0, 0])
            audit = check_value_claims(text, log)
            for c in audit["claims"]:
                kinds[c["kind"]][0] += int(c["match_true"])
                kinds[c["kind"]][1] += 1
            allowed = CITE_ONLY_FIELDS if cond == "cite_only" else None
            rows[(model, seed, cond)].append({
                "claims": audit["n_claims"],
                "kinds": dict(kinds),
                "ur": score_explanation(text, log, allowed_fields=allowed)["unsupported_rate"],
                "ur_split": score_explanation(split_citations(text), log, allowed_fields=allowed)["unsupported_rate"],
            })


def seed_value(items, metric):
    if metric in ("ur", "ur_split", "claims"):
        return float(np.mean([r[metric] for r in items]))
    hit = sum(r["kinds"].get(metric, [0, 0])[0] for r in items)
    tot = sum(r["kinds"].get(metric, [0, 0])[1] for r in items)
    return hit / tot if tot else float("nan")


def contrast(table, metric, frm, to):
    d = table[to][metric] - table[frm][metric]
    d = d[~np.isnan(d)]
    p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d) else float("nan")
    return {"from": frm, "to": to, "delta": float(np.mean(d)), "p": p, "seeds_up": int((d > 0).sum()), "n": len(d)}


def mean(values):
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else None


conds = ["grounded", "evidence_only", "cite_only", "free"]
metrics = ["number", "prominence", "claims", "ur", "ur_split"]
out = {}
for model in sorted({m for m, _, _ in rows}):
    seeds = sorted({s for m, s, _ in rows if m == model and all((m, s, k) in rows for k in conds)})
    table = {c: {mt: np.array([seed_value(rows[(model, s, c)], mt) for s in seeds]) for mt in metrics} for c in conds}
    res = {"n_seeds": len(seeds), **{c: {m: mean(table[c][m]) for m in metrics} for c in conds}}
    for ur in ("ur", "ur_split"):
        total = res["free"][ur] - res["grounded"][ur]
        for name, (frm, to) in {"contract_alone": ("free", "cite_only"), "evidence_given_contract": ("cite_only", "grounded")}.items():
            res[f"{ur}_{name}"] = {**contrast(table, ur, frm, to), "share_of_total": (res[frm][ur] - res[to][ur]) / total}
    res["claims_simple"] = {
        "contract_given_evidence": contrast(table, "claims", "evidence_only", "grounded"),
        "contract_without_evidence": contrast(table, "claims", "free", "cite_only"),
        "evidence_given_contract": contrast(table, "claims", "cite_only", "grounded"),
        "evidence_without_contract": contrast(table, "claims", "free", "evidence_only"),
    }
    res["prominence_contract_given_evidence"] = contrast(table, "prominence", "evidence_only", "grounded")
    out[model] = res

    print(f"\n== {model}  (seeds={len(seeds)})")
    print(f"{'condition':14s} " + " ".join(f"{m:>11s}" for m in metrics))
    for c in conds:
        print(f"{c:14s} " + " ".join(f"{res[c][m]:11.3f}" if res[c][m] is not None else f"{'--':>11s}" for m in metrics))
    for ur in ("ur", "ur_split"):
        a, b = res[f"{ur}_contract_alone"], res[f"{ur}_evidence_given_contract"]
        print(f"  {ur:8s} contract alone {a['delta']:+.3f} ({a['share_of_total']:.2f})  evidence|contract {b['delta']:+.3f} ({b['share_of_total']:.2f})  p={a['p']:.2g}/{b['p']:.2g}")
    for name, v in res["claims_simple"].items():
        print(f"  claims {name:26s} {v['delta']:+.3f} p={v['p']:.2g} ({v['seeds_up']}/{v['n']} up)")
    v = res["prominence_contract_given_evidence"]
    print(f"  prominence contract|evidence     {v['delta']:+.3f} p={v['p']:.2g} ({v['seeds_up']}/{v['n']} up)")

path = pathlib.Path("results/post_freeze") / (f"two_by_two_{SINGLE.name}.json" if SINGLE else "two_by_two.json")
path.write_text(json.dumps(out, indent=2))
print(f"\nwritten {path}")

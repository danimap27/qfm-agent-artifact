"""Human validation of the citation auditor: sample, blind sheet, and agreement.

    python scripts/annotation_package.py sample   # writes the blind sheet and the key
    python scripts/annotation_package.py kappa A.csv B.csv

``sample`` draws 300 sentences that carry at least one citation, 75 from each
condition that cites (grounded, sabotage, occlusion, contract-only), balanced
over the three verbalizers and over the auditor's two verdicts, and writes a blind CSV for annotators with the
sentence and the log fields it was allowed to cite, plus a separate key with
the auditor's verdict. ``kappa`` reads two filled sheets and reports Cohen's
kappa between the annotators and between each annotator and the auditor, with
a bootstrap 95 percent interval over sentences.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.eval.claims import score_explanation
from code.evidence.manipulations import occlude_log, sabotage_log
from code.verbalizer.prompts_2x2 import CITE_ONLY_FIELDS

OUT = pathlib.Path("results/post_freeze/annotation")
CONDITIONS = {"grounded": "grid_q1", "sabotage": "grid_q1", "occlusion_attributions": "grid_q1", "cite_only": "grid_2x2"}
MODELS = ("qwen35", "llama31_8b", "gemma3_27b")
PER_CONDITION = 75


def shown_log(condition: str, log: dict) -> dict:
    if condition == "sabotage":
        return sabotage_log(log)
    if condition == "occlusion_attributions":
        return occlude_log(log, ["attributions"])
    if condition == "cite_only":
        return {"prediction": {k: log["prediction"][k] for k in ("class_index", "probability")}}
    return log


def evidence_text(log: dict) -> str:
    """The citable fields as the annotator sees them, without bookkeeping."""
    keep = {k: log[k] for k in ("prediction", "attributions", "counterfactuals") if k in log}
    return json.dumps(keep, indent=1, sort_keys=True)


def sample() -> None:
    rng = random.Random(20261021)
    rows, key, used = [], [], set()
    for condition, sub in CONDITIONS.items():
        pool = []
        for model in MODELS:
            for cell in sorted(pathlib.Path("results", sub, model).glob("*/seed*")):
                logs = {json.loads(l)["request_id"]: json.loads(l) for l in open(cell / "logs.jsonl") if l.strip()}
                for line in open(cell / "explanations.jsonl"):
                    rec = json.loads(line)
                    if rec["condition"] != condition:
                        continue
                    shown = shown_log(condition, logs[rec["request_id"]])
                    allowed = CITE_ONLY_FIELDS if condition == "cite_only" else None
                    text = rec["parsed"].get("explanation") or rec["response"]
                    # the contract-only view hides fields, so it is scored on the full log with the whitelist
                    scored = logs[rec["request_id"]] if condition == "cite_only" else shown
                    for s in score_explanation(text, scored, allowed_fields=allowed)["sentences"]:
                        if s["citations"]:
                            pool.append((model, cell, rec["request_id"], s, shown))
        # half from each auditor verdict, so agreement is measured on both classes
        strata = [[p for p in pool if p[0] == m and (p[3]["status"] == "supported") == ok] for m in MODELS for ok in (True, False)]
        for i in range(PER_CONDITION):
            stratum = strata[i % len(strata)] or [p for p in pool if p[0] == MODELS[i % len(MODELS)]]
            model, cell, request_id, s, shown = rng.choice(stratum)
            while (s["sentence"], evidence_text(shown)) in used:
                model, cell, request_id, s, shown = rng.choice(stratum)
            used.add((s["sentence"], evidence_text(shown)))
            item = f"{condition[:3]}-{i:03d}"
            rows.append({"item": item, "sentence": s["sentence"], "evidence": evidence_text(shown),
                         "supported (1 yes / 0 no)": "", "notes": ""})
            key.append({"item": item, "condition": condition, "model": model, "cell": str(cell),
                        "request_id": request_id, "auditor_supported": int(s["status"] == "supported")})
    rng.shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sheet_blind.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "key.jsonl").write_text("\n".join(json.dumps(k) for k in key) + "\n")
    print(f"wrote {len(rows)} items to {OUT}/sheet_blind.csv and the key to {OUT}/key.jsonl")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def read_labels(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8-sig") as handle:
        return {r["item"]: int(r["supported (1 yes / 0 no)"]) for r in csv.DictReader(handle)
                if r["supported (1 yes / 0 no)"].strip() in ("0", "1")}


def kappa(path_a: str, path_b: str) -> None:
    a, b = read_labels(path_a), read_labels(path_b)
    auditor = {k["item"]: k["auditor_supported"] for k in map(json.loads, open(OUT / "key.jsonl"))}
    items = sorted(set(a) & set(b) & set(auditor))
    va, vb, vk = (np.array([d[i] for i in items]) for d in (a, b, auditor))
    rng = np.random.default_rng(0)
    for name, x, y in (("annotator A vs B", va, vb), ("annotator A vs auditor", va, vk), ("annotator B vs auditor", vb, vk)):
        boot = [cohen_kappa(x[idx], y[idx]) for idx in (rng.integers(0, len(items), len(items)) for _ in range(2000))]
        print(f"{name:24s} kappa={cohen_kappa(x, y):.3f} 95% CI [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}] n={len(items)}")


assert abs(cohen_kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) - 1.0) < 1e-12
assert abs(cohen_kappa(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))) < 1e-12

if __name__ == "__main__":
    {"sample": lambda: sample(), "kappa": lambda: kappa(sys.argv[2], sys.argv[3])}[sys.argv[1]]()

"""Re-score every grid explanation with the value-based auditor.

Walks the per-cell caches of the frozen grid, scores numeric, class and
prominence claims against the true log (and against the shown log for the
tamper sensitivity), and aggregates per condition with template averaging.
Output: results/post_freeze/value_audit.json plus a printed summary.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.eval.value_audit import check_value_claims

ROOT = pathlib.Path("results/grid_q1")


def build_log_index(cell_dir: pathlib.Path) -> dict:
    logs = {}
    path = cell_dir / "logs.jsonl"
    if not path.exists():
        return logs
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            log = json.loads(line)
            logs[log["request_id"]] = log
    return logs


def null_partner(logs: dict) -> dict:
    """Map each request to the next request of the same task in the same cell.

    Scoring an explanation against another request's log gives the empirical
    chance agreement of the auditor, which replaces the analytic floor.
    """
    by_task = defaultdict(list)
    for request_id in sorted(logs):
        by_task[request_id.rsplit("-", 1)[0]].append(request_id)
    partner = {}
    for ids in by_task.values():
        for i, request_id in enumerate(ids):
            partner[request_id] = logs[ids[(i + 1) % len(ids)]]
    return partner


rows = []
for cell_dir in sorted(ROOT.glob("*/*/*")):
    explanations = cell_dir / "explanations.jsonl"
    if not explanations.exists():
        continue
    logs = build_log_index(cell_dir)
    partners = null_partner(logs)
    for line in explanations.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        log = logs.get(record["request_id"])
        if log is None:
            continue
        condition = record["condition"]
        text = record["parsed"].get("explanation") or record["response"]
        # the log as shown to the verbalizer, for tamper sensitivity
        if condition == "sabotage":
            from code.evidence.log_builder import sabotage_log

            shown = sabotage_log(log)
        elif condition == "occlusion_attributions":
            from code.evidence.log_builder import occlude_log

            shown = occlude_log(log, ["attributions"])
        else:
            shown = log
        result = check_value_claims(text, log, shown)
        null = check_value_claims(text, partners[record["request_id"]])
        by_kind: dict = defaultdict(lambda: [0, 0, 0, 0])
        for claim, null_claim in zip(result["claims"], null["claims"]):
            entry = by_kind[claim["kind"]]
            entry[0] += int(claim["match_true"])
            entry[1] += int(claim["match_shown"])
            entry[2] += 1
            entry[3] += int(null_claim["match_true"])
        rows.append(
            {
                "condition": condition,
                "model": record.get("llm_model"),
                "template": record.get("template"),
                "n_claims": result["n_claims"],
                "match_true": result["n_match_true"],
                "match_shown": result["n_match_shown"],
                "by_kind": {k: list(v) for k, v in by_kind.items()},
            }
        )

print(f"scored {len(rows)} explanations")

# per condition: template-averaged accuracy over (model, seed, request) units
acc_true: dict = defaultdict(list)
acc_shown: dict = defaultdict(list)
pairs = defaultdict(lambda: [0, 0, 0])  # condition -> [match_true, match_shown, n_claims]
for row in rows:
    if row["n_claims"] == 0:
        continue
    pairs[row["condition"]][0] += row["match_true"]
    pairs[row["condition"]][1] += row["match_shown"]
    pairs[row["condition"]][2] += row["n_claims"]

summary = {}
for condition, (m_true, m_shown, n) in sorted(pairs.items()):
    summary[condition] = {
        "n_claims": n,
        "value_accuracy_true": m_true / n,
        "value_accuracy_shown": m_shown / n,
    }
    print(
        f"{condition:24s} claims={n:7d} accuracy_true={m_true / n:.3f} "
        f"accuracy_shown={m_shown / n:.3f}"
    )

kind_pairs: dict = defaultdict(lambda: [0, 0, 0, 0])  # [true, shown, n, null]
for row in rows:
    for kind, (m_true, m_shown, n, m_null) in row["by_kind"].items():
        bucket = kind_pairs[(row["condition"], kind)]
        bucket[0] += m_true
        bucket[1] += m_shown
        bucket[2] += n
        bucket[3] += m_null
print()
for (condition, kind), (m_true, m_shown, n, m_null) in sorted(kind_pairs.items()):
    if n == 0:
        continue
    print(f"{condition:24s} {kind:10s} n={n:6d} true={m_true / n:.3f} shown={m_shown / n:.3f} null={m_null / n:.3f}")

out = ROOT.parent / "post_freeze" / "value_audit.json"
out.write_text(json.dumps({"summary": summary, "by_kind": {f"{c}|{k}": v for (c, k), v in kind_pairs.items()}, "n_explanations": len(rows)}, indent=2), encoding="utf-8")
print(f"written {out}")

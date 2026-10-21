"""Value-based claim verification independent of the citation contract.

The unsupported rate is conditioned on citations, so the free condition
attains 1.000 by construction. This auditor scores every numeric, class and
prominence claim against the log values themselves, whether or not the
sentence cites anything, which makes the metric fair across all conditions.
For the tamper condition the same claims are additionally scored against the
log as shown to the verbalizer, which measures whether the explanation
follows the manipulated record or the true one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from code.eval.claims import CLASS_RE, FEATURE_RE, NUMBER_RE, SUPERLATIVE_RE, _allowed_numbers, split_sentences

PERCENT_AFTER_RE = re.compile(r"\s*(?:%|percent)", re.IGNORECASE)


def _matches(number: float, allowed: List[float]) -> bool:
    # ponytail: matches against every number in the log with the citation
    # audit's tolerance, so chance agreement is high (measured against a
    # mismatched log in scripts/value_audit_grid.py: ~0.59 numbers, ~0.80
    # classes); report it next to the metric, field-typed matching lowers it.
    return any(abs(number - value) <= max(0.05, 0.05 * abs(value)) for value in allowed)


def check_value_claims(
    text: str,
    true_log: Dict[str, Any],
    shown_log: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score numeric, class and prominence claims against log values.

    Percentages are read as fractions (85% is checked as 0.85). A prominence
    claim matches a log only if that log has attributions and the claimed
    feature is its top-ranked one, so an occluded log never vacuously agrees.
    """
    shown_log = true_log if shown_log is None else shown_log
    true_numbers = _allowed_numbers(true_log)
    shown_numbers = _allowed_numbers(shown_log)
    true_class = int(true_log["prediction"]["class_index"])
    shown_class = int(shown_log["prediction"]["class_index"])
    true_top = (true_log["attributions"][0]["index"] + 1) if true_log.get("attributions") else None
    shown_top = (shown_log["attributions"][0]["index"] + 1) if shown_log.get("attributions") else None
    flip_targets = {int(cf["flips_to_index"]) for cf in true_log.get("counterfactuals", [])}
    flip_targets |= {int(cf["flips_to_index"]) for cf in shown_log.get("counterfactuals", [])}

    claims: List[Dict[str, Any]] = []
    for sentence in split_sentences(text):
        for match in NUMBER_RE.finditer(sentence):
            number = float(match.group(0))
            percent = bool(PERCENT_AFTER_RE.match(sentence, match.end()))
            if not percent and abs(number - round(number)) < 1e-9 and abs(number) <= 10:
                continue  # class-like integers are handled below
            if percent:
                number /= 100.0
            claims.append(
                {
                    "kind": "number",
                    "value": number,
                    "match_true": _matches(number, true_numbers),
                    "match_shown": _matches(number, shown_numbers),
                }
            )
        for match in CLASS_RE.finditer(sentence):
            claimed = int(match.group(1))
            claims.append(
                {
                    "kind": "class",
                    "value": claimed,
                    "match_true": claimed == true_class or claimed in flip_targets,
                    "match_shown": claimed == shown_class or claimed in flip_targets,
                }
            )
        if SUPERLATIVE_RE.search(sentence):
            for match in FEATURE_RE.finditer(sentence):
                claimed = int(match.group(1))
                claims.append(
                    {
                        "kind": "prominence",
                        "value": claimed,
                        "match_true": claimed == true_top,
                        "match_shown": claimed == shown_top,
                    }
                )

    n = len(claims)
    n_true = sum(1 for claim in claims if claim["match_true"])
    n_shown = sum(1 for claim in claims if claim["match_shown"])
    return {
        "n_claims": n,
        "n_match_true": n_true,
        "n_match_shown": n_shown,
        "value_accuracy_true": (n_true / n) if n else None,
        "value_accuracy_shown": (n_shown / n) if n else None,
        "claims": claims,
    }

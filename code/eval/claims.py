"""Claim extraction and automatic scoring for explanations (scorer v1).

The scorer works at the sentence level. For citation-carrying conditions it
checks that every sentence cites an existing log field, that numbers match the
cited fields, and that feature and class references agree with the cited rows.
Citations may name a whole log section (for example [attributions]) when the
model does not pick the most specific identifier, in which case the section is
expanded and the values of every row in it are allowed. Counterfactual
sentences are parsed into executable edits, one or several per sentence, so the
fidelity harness can re-run them on the substrate, which checks consistency by
execution instead of by reading. The scorer is deliberately conservative and
heuristic: the protocol reserves a human-audited sample to bound its error.
"""

from __future__ import annotations

import re
from typing import Any, Optional

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\-]+)\]")
BRACKET_ID_RE = re.compile(r"\[\s*[A-Za-z_.]*\bid\s*=\s*[\"']?([A-Za-z0-9_.\-]+)[\"']?\s*\]")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FEATURE_RE = re.compile(r"feature\s+(\d+)", re.IGNORECASE)
CLASS_RE = re.compile(r"class\s+(\d+)", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CF_FROM_TO_RE = re.compile(
    r"feature\s+(\d+)\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
CF_TO_RE = re.compile(r"feature\s+(\d+)\b[^.]*?\bto\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps abbreviations simple on purpose."""
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def field_table(log: dict) -> dict[str, Any]:
    """Flat map from citable field identifiers to their values.

    Section aliases (prediction, attributions, counterfactuals, ood, routing,
    request) are included so that a citation of a whole section stays checkable
    against the rows it contains.
    """
    prediction = log["prediction"]
    attributions = log.get("attributions", [])
    counterfactuals = log.get("counterfactuals", [])
    ood = log.get("ood", {})
    routing = log.get("routing") or {}
    request = log.get("request", {})
    table: dict[str, Any] = {
        "prediction": prediction,
        "prediction.label": prediction["label"],
        "prediction.class_index": prediction["class_index"],
        "prediction.probability": prediction["probability"],
        "prediction.calibrated": prediction["calibrated"],
        "prediction.head_accuracy_on_task": prediction["head_accuracy_on_task"],
        "attributions": attributions,
        "counterfactuals": counterfactuals,
        "ood": ood,
        "ood.flag": ood["flag"],
        "ood.score": ood["score"],
        "routing": routing,
        "routing.decision": routing.get("decision"),
        "routing.confidence": routing.get("confidence"),
        "routing.method": routing.get("method"),
        "request": request,
        "request.text": request.get("text"),
    }
    for row in attributions:
        table[f"attributions.rank{row['rank']}"] = row
    for cf in counterfactuals:
        table[cf["id"]] = cf
    return table


def _allowed_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(match) for match in NUMBER_RE.findall(value)]
    if isinstance(value, dict):
        numbers: list[float] = []
        for item in value.values():
            numbers.extend(_allowed_numbers(item))
        return numbers
    if isinstance(value, (list, tuple)):
        numbers = []
        for item in value:
            numbers.extend(_allowed_numbers(item))
        return numbers
    return []


def _allowed_classes(value: Any) -> set[int]:
    classes: set[int] = set()
    if isinstance(value, dict):
        if "class_index" in value:
            classes.add(int(value["class_index"]))
        if "flips_to_index" in value:
            classes.add(int(value["flips_to_index"]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            classes |= _allowed_classes(item)
    return classes


def _allowed_features(value: Any) -> set[int]:
    features: set[int] = set()
    if isinstance(value, dict):
        if "index" in value:
            features.add(int(value["index"]) + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            features |= _allowed_features(item)
    return features


def score_sentence(sentence: str, table: dict[str, Any]) -> dict:
    """Classify one sentence as supported or unsupported by its cited fields."""
    sentence = BRACKET_ID_RE.sub(r"[\1]", sentence)
    citations = CITATION_RE.findall(sentence)
    if not citations:
        return {"status": "uncited", "citations": []}
    unknown = [name for name in citations if name not in table]
    if unknown:
        return {"status": "invalid_citation", "citations": citations, "unknown": unknown}
    cited = [table[name] for name in citations]
    numbers = [float(match) for match in NUMBER_RE.findall(sentence)]
    allowed_numbers = [number for value in cited for number in _allowed_numbers(value)]
    for number in numbers:
        if abs(number - round(number)) < 1e-9 and abs(number) <= 10:
            continue
        if not any(abs(number - allowed) <= max(0.05, 0.05 * abs(allowed)) for allowed in allowed_numbers):
            return {"status": "value_mismatch", "citations": citations}
    classes = {int(match) for match in CLASS_RE.findall(sentence)}
    if classes:
        base_class = table.get("prediction.class_index")
        allowed_classes: set[int] = {int(base_class)} if base_class is not None else set()
        for value in cited:
            allowed_classes |= _allowed_classes(value)
        if not classes.issubset(allowed_classes):
            return {"status": "value_mismatch", "citations": citations}
    features = {int(match) for match in FEATURE_RE.findall(sentence)}
    if features:
        allowed_features: set[int] = set()
        for value in cited:
            allowed_features |= _allowed_features(value)
        if allowed_features and not features.issubset(allowed_features):
            return {"status": "value_mismatch", "citations": citations}
    return {"status": "supported", "citations": citations}


def parse_counterfactuals(sentence: str) -> list[dict]:
    """Every executable edit described by a sentence, when parseable.

    A single sentence can describe several counterfactual changes: each match
    becomes its own claim, and the class mentioned after the match, if any, is
    taken as that claim's target class.
    """
    if not FEATURE_RE.search(sentence):
        return []
    matches: list[tuple[int, int, int, Optional[float], float]] = []
    for match in CF_FROM_TO_RE.finditer(sentence):
        matches.append(
            (match.start(), match.end(), int(match.group(1)) - 1, float(match.group(2)), float(match.group(3)))
        )
    spans = [(start, end) for start, end, *_ in matches]
    for match in CF_TO_RE.finditer(sentence):
        if any(start <= match.start() < end for start, end in spans):
            continue
        matches.append((match.start(), match.end(), int(match.group(1)) - 1, None, float(match.group(2))))
    matches.sort(key=lambda item: item[0])
    claims: list[dict] = []
    for _, end, index, from_value, to_value in matches:
        class_match = CLASS_RE.search(sentence, end)
        claims.append(
            {
                "index": index,
                "from_value": from_value,
                "to_value": to_value,
                "target_class": int(class_match.group(1)) if class_match else None,
                "sentence": sentence,
            }
        )
    return claims


def parse_counterfactual(sentence: str) -> Optional[dict]:
    """First executable edit described by a sentence, when parseable."""
    claims = parse_counterfactuals(sentence)
    return claims[0] if claims else None


def score_explanation(text: str, log: dict, allowed_fields: Optional[tuple] = None) -> dict:
    """Sentence-level scoring of one explanation against one decision log.

    ``allowed_fields`` restricts the citable field names to the ones the
    condition was actually shown, so citing hidden fields is invalid.
    """
    table = field_table(log)
    if allowed_fields is not None:
        table = {name: value for name, value in table.items() if name in allowed_fields}
    sentences = split_sentences(text)
    scored: list[dict] = []
    for sentence in sentences:
        result = score_sentence(sentence, table)
        result["sentence"] = sentence
        claims = parse_counterfactuals(sentence)
        if claims:
            result["counterfactuals"] = claims
        scored.append(result)
    unsupported = [item for item in scored if item["status"] != "supported"]
    features = sorted({int(match) for item in scored for match in FEATURE_RE.findall(item["sentence"])})
    return {
        "n_sentences": len(scored),
        "n_unsupported": len(unsupported),
        "unsupported_rate": (len(unsupported) / len(scored)) if scored else 1.0,
        "sentences": scored,
        "feature_mentions": features,
        "counterfactual_claims": [claim for item in scored for claim in item.get("counterfactuals", [])],
    }


CONFIDENCE_WORDS = re.compile(r"probab|confid|likelihood", re.IGNORECASE)
PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
SUPERLATIVE_RE = re.compile(
    r"most important|primary|largest|dominant|main driver|drives|driven", re.IGNORECASE
)


def check_contradictions(text: str, log: dict) -> dict:
    """Lightweight contradiction audit against the log, without citations.

    Counts checkable assertions (probability magnitudes, class references,
    feature prominence claims) and how many contradict the true log. This is
    the cross-condition counterpart of the citation-based unsupported rate,
    since free and single-block explanations never cite. Heuristic by design,
    like the rest of the scorer.
    """
    true_probability = log["prediction"]["probability"]
    true_class = log["prediction"]["class_index"]
    flip_targets = {cf["flips_to_index"] for cf in log.get("counterfactuals", [])}
    top1 = (log["attributions"][0]["index"] + 1) if log.get("attributions") else None

    checks = 0
    contradictions: list[dict] = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        probability = None
        percent = PERCENT_RE.search(sentence)
        if percent:
            value = float(percent.group(1)) / 100.0
            if 0.0 < value <= 1.0:
                probability = value
        if probability is None and CONFIDENCE_WORDS.search(lowered):
            for number in NUMBER_RE.findall(sentence):
                value = float(number)
                if 0.0 < value <= 1.0:
                    probability = value
                    break
        if probability is not None and true_probability is not None:
            checks += 1
            if abs(probability - float(true_probability)) > 0.05:
                contradictions.append(
                    {
                        "kind": "probability",
                        "claimed": probability,
                        "actual": float(true_probability),
                        "sentence": sentence,
                    }
                )
        for match in CLASS_RE.finditer(sentence):
            claimed_class = int(match.group(1))
            checks += 1
            if claimed_class != true_class and claimed_class not in flip_targets:
                contradictions.append(
                    {"kind": "class", "claimed": claimed_class, "actual": true_class, "sentence": sentence}
                )
        if top1 is not None and SUPERLATIVE_RE.search(lowered):
            for match in FEATURE_RE.finditer(sentence):
                claimed_feature = int(match.group(1))
                checks += 1
                if claimed_feature != top1:
                    contradictions.append(
                        {
                            "kind": "feature_prominence",
                            "claimed": claimed_feature,
                            "actual": top1,
                            "sentence": sentence,
                        }
                    )
    n_sentences = len(split_sentences(text))
    return {
        "n_checks": checks,
        "n_contradictions": len(contradictions),
        "contradiction_rate": (len(contradictions) / checks) if checks else None,
        "checkable_coverage": (checks / n_sentences) if n_sentences else 0.0,
        "contradictions": contradictions,
    }

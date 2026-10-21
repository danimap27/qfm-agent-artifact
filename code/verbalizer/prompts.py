"""Prompt templates and per-condition message builders (template version v3).

Templates t1 to t3 present the same information under the same citation
contract and differ only in how the explanation is packaged: t1 is the frozen
table-style prompt used by every run before the Q1 grid, t2 frames the answer
as a narrative paragraph, and t3 as a numbered checklist. The t1 messages are
byte-identical to template version v2, so every cached prompt hash from
earlier runs stays valid. The single-block prompt keeps the fixed
Task/Class/Explanation parse contract in all three templates.
"""

from __future__ import annotations

import hashlib
import json

PROMPT_VERSION = "v3"

TEMPLATES = ("t1", "t2", "t3")

SYSTEM_GROUNDED = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log "
    "whose entries carry stable identifiers in square brackets. Write an explanation of the "
    "prediction for the user in two to four sentences. Every claim must end with the citation of "
    "the log field that supports it between square brackets, using the most specific identifier "
    "listed in the log, for example [prediction.probability], [attributions.rank1], or [cf-1]. "
    "Do not state "
    "anything that is not present in the log. If the log contains a counterfactual entry whose "
    "identifier starts with cf-, describe its change in one sentence, naming the feature and the "
    "value it changes to, for example changing feature 3 to 0.40 would move the prediction to "
    "class 1, and cite that entry. Keep the explanation under 120 words."
)

SYSTEM_GROUNDED_T2 = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log "
    "whose entries carry stable identifiers in square brackets. Write the explanation for the "
    "user as a short narrative paragraph of two to four sentences. Every claim must end with the "
    "citation of "
    "the log field that supports it between square brackets, using the most specific identifier "
    "listed in the log, for example [prediction.probability], [attributions.rank1], or [cf-1]. "
    "Do not state "
    "anything that is not present in the log. If the log contains a counterfactual entry whose "
    "identifier starts with cf-, weave its change into the narrative, naming the feature and the "
    "value it changes to, for example changing feature 3 to 0.40 would move the prediction to "
    "class 1, and cite that entry. Keep the explanation under 120 words."
)

SYSTEM_GROUNDED_T3 = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log "
    "whose entries carry stable identifiers in square brackets. Write the explanation for the "
    "user as a numbered checklist with two to four items. Every item must end with the citation of "
    "the log field that supports it between square brackets, using the most specific identifier "
    "listed in the log, for example [prediction.probability], [attributions.rank1], or [cf-1]. "
    "Do not state "
    "anything that is not present in the log. If the log contains a counterfactual entry whose "
    "identifier starts with cf-, describe its change as the final checklist item, naming the "
    "feature and the value it changes to, for example changing feature 3 to 0.40 would move the "
    "prediction to class 1, and cite that entry. Keep the explanation under 120 words."
)

SYSTEM_FREE = (
    "You are an expert assistant. Explain the prediction for the user in two to four sentences, "
    "in clear and specific language. Keep the explanation under 100 words."
)

SYSTEM_FREE_T2 = (
    "You are an expert assistant. Explain the prediction for the user as a short narrative "
    "paragraph of two to four sentences, in clear and specific language. Keep the explanation "
    "under 100 words."
)

SYSTEM_FREE_T3 = (
    "You are an expert assistant. Explain the prediction for the user as a numbered checklist of "
    "two to four items, in clear and specific language. Keep the explanation under 100 words."
)

SYSTEM_SINGLE_BLOCK = (
    "You are a complete prediction pipeline. Given a request and a numeric feature payload, decide "
    "which task it belongs to, predict the class, and then explain the prediction to the user. "
    "The candidate tasks are: {candidates}. "
    "Answer exactly in this format:\nTask: <task identifier>\nClass: <class name or class index>\n"
    "Explanation: <two to four sentences, under 100 words>"
)

SYSTEM_SINGLE_BLOCK_T2 = (
    "You are a complete prediction pipeline. Given a request and a numeric feature payload, decide "
    "which task it belongs to, predict the class, and then explain the prediction to the user. "
    "The candidate tasks are: {candidates}. "
    "Start your answer with the two fixed header lines below, then give the explanation as a "
    "short narrative paragraph. Answer exactly in this format:\n"
    "Task: <task identifier>\nClass: <class name or class index>\n"
    "Explanation: <two to four sentences, under 100 words>"
)

SYSTEM_SINGLE_BLOCK_T3 = (
    "You are a complete prediction pipeline. Given a request and a numeric feature payload, decide "
    "which task it belongs to, predict the class, and then explain the prediction to the user. "
    "The candidate tasks are: {candidates}. "
    "Start your answer with the two fixed header lines below, then give the explanation as a "
    "numbered checklist. Answer exactly in this format:\n"
    "Task: <task identifier>\nClass: <class name or class index>\n"
    "Explanation: <two to four numbered items, under 100 words>"
)

_SYSTEMS_GROUNDED = {"t1": SYSTEM_GROUNDED, "t2": SYSTEM_GROUNDED_T2, "t3": SYSTEM_GROUNDED_T3}
_SYSTEMS_FREE = {"t1": SYSTEM_FREE, "t2": SYSTEM_FREE_T2, "t3": SYSTEM_FREE_T3}
_SYSTEMS_SINGLE_BLOCK = {"t1": SYSTEM_SINGLE_BLOCK, "t2": SYSTEM_SINGLE_BLOCK_T2, "t3": SYSTEM_SINGLE_BLOCK_T3}


def _check_template(template: str) -> None:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template}")


def prompt_hash(messages: list[dict]) -> str:
    """Short stable hash of a message list, recorded next to every explanation."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def _prompt_evidence(evidence: dict) -> dict:
    """Evidence exactly as serialized into prompts, volatile fields excluded.

    The log keeps its creation timestamp for provenance, but the serialization
    that reaches the language model must not carry wall-clock fields, because
    they would change the message hash on every rebuild and break replay.
    """
    clean = dict(evidence)
    meta = clean.get("meta")
    if isinstance(meta, dict):
        clean["meta"] = {key: value for key, value in meta.items() if key != "created"}
    return clean


def grounded_messages(request_text: str, evidence: dict, template: str = "t1") -> list[dict]:
    """Grounded condition: full (or manipulated) log plus the citation contract."""
    _check_template(template)
    body = json.dumps(_prompt_evidence(evidence), sort_keys=True)
    return [
        {"role": "system", "content": _SYSTEMS_GROUNDED[template]},
        {"role": "user", "content": f"Request: {request_text}\n\nDecision log:\n{body}\n\nWrite the explanation now."},
    ]


def free_messages(request_text: str, prediction: str, probability: float, template: str = "t1") -> list[dict]:
    """Free condition: prediction and probability only."""
    _check_template(template)
    return [
        {"role": "system", "content": _SYSTEMS_FREE[template]},
        {
            "role": "user",
            "content": f"Request: {request_text}\n\nPrediction: {prediction} with probability {probability:.2f}.\n\nExplain the prediction.",
        },
    ]


def single_block_messages(
    request_text: str, features: list[float], task_ids: list[str], template: str = "t1"
) -> list[dict]:
    """Single-block condition: the model routes, predicts, and explains in one pass."""
    _check_template(template)
    payload = ", ".join(f"{value:.2f}" for value in features)
    candidates = ", ".join(task_ids)
    return [
        {"role": "system", "content": _SYSTEMS_SINGLE_BLOCK[template].format(candidates=candidates)},
        {
            "role": "user",
            "content": f"Request: {request_text}\nFeature payload (8 values): [{payload}]\n\nRespond in the exact format.",
        },
    ]


def template_floor_text(log: dict) -> str:
    """Deterministic explanation built from the log without a language model."""
    prediction = log["prediction"]
    parts = [
        f"The prediction is {prediction['label']} with probability {prediction['probability']:.2f} "
        "[prediction.probability]."
    ]
    if log["attributions"]:
        top = log["attributions"][0]
        parts.append(
            f"The most influential feature is {top['feature']} with a prediction change of "
            f"{top['delta']:.2f} [attributions.rank1]."
        )
    if log["counterfactuals"]:
        cf = log["counterfactuals"][0]
        parts.append(
            f"Changing {cf['feature']} from {cf['from']:.2f} to {cf['to']:.2f} would change the "
            f"prediction to {cf['flips_to']} [{cf['id']}]."
        )
    return " ".join(parts)

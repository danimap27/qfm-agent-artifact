"""2x2 condition builders: evidence and the citation contract crossed.

Cell definitions of the evidence-by-contract design:
- grounded: evidence yes, contract yes (existing).
- evidence_only: evidence yes, contract no (this file).
- cite_only: evidence no, contract yes (this file).
- free: evidence no, contract no (existing).
"""

from __future__ import annotations

import json
from typing import Dict, List

from code.verbalizer.prompts import _check_template, _prompt_evidence

# The only fields the contract-only condition is shown, and so may cite.
CITE_ONLY_FIELDS = ("prediction.class_index", "prediction.probability")

SYSTEM_EVIDENCE_ONLY = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log. "
    "Write an explanation of the prediction for the user in two to four sentences, in clear and "
    "specific language. Do not state anything that is not present in the log. If the log contains "
    "a counterfactual entry whose identifier starts with cf-, describe its change in one sentence, "
    "naming the feature and the value it changes to, for example changing feature 3 to 0.40 would "
    "move the prediction to class 1. Keep the explanation under 120 words."
)

SYSTEM_EVIDENCE_ONLY_T2 = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log. "
    "Write the explanation for the user as a short narrative paragraph of two to four sentences. "
    "Do not state anything that is not present in the log. If the log contains a counterfactual "
    "entry whose identifier starts with cf-, weave its change into the narrative, naming the "
    "feature and the value it changes to, for example changing feature 3 to 0.40 would move the "
    "prediction to class 1. Keep the explanation under 120 words."
)

SYSTEM_EVIDENCE_ONLY_T3 = (
    "You are a faithful explainer for an automated prediction system. You receive a decision log. "
    "Write the explanation for the user as a numbered checklist with two to four items. Do not "
    "state anything that is not present in the log. If the log contains a counterfactual entry "
    "whose identifier starts with cf-, describe its change as the final checklist item, naming the "
    "feature and the value it changes to, for example changing feature 3 to 0.40 would move the "
    "prediction to class 1. Keep the explanation under 120 words."
)

_CITE_TAIL = (
    "using [prediction.class_index] or [prediction.probability]. "
    "Keep the explanation under 100 words."
)

SYSTEM_CITE_ONLY = (
    "You are an expert assistant. Explain the prediction for the user in two to four sentences, "
    "in clear and specific language. Every claim must end with the citation of the log field that "
    "supports it between square brackets, " + _CITE_TAIL
)

SYSTEM_CITE_ONLY_T2 = (
    "You are an expert assistant. Explain the prediction for the user as a short narrative "
    "paragraph of two to four sentences, in clear and specific language. Every claim must end "
    "with the citation of the log field that supports it between square brackets, " + _CITE_TAIL
)

SYSTEM_CITE_ONLY_T3 = (
    "You are an expert assistant. Explain the prediction for the user as a numbered checklist of "
    "two to four items, in clear and specific language. Every item must end with the citation of "
    "the log field that supports it between square brackets, " + _CITE_TAIL
)

_SYSTEMS_EVIDENCE_ONLY: Dict[str, str] = {
    "t1": SYSTEM_EVIDENCE_ONLY,
    "t2": SYSTEM_EVIDENCE_ONLY_T2,
    "t3": SYSTEM_EVIDENCE_ONLY_T3,
}
_SYSTEMS_CITE_ONLY: Dict[str, str] = {
    "t1": SYSTEM_CITE_ONLY,
    "t2": SYSTEM_CITE_ONLY_T2,
    "t3": SYSTEM_CITE_ONLY_T3,
}


def evidence_only_messages(request_text: str, evidence: dict, template: str = "t1") -> List[dict]:
    """Evidence without the citation contract."""
    _check_template(template)
    body = json.dumps(_prompt_evidence(evidence), sort_keys=True)
    return [
        {"role": "system", "content": _SYSTEMS_EVIDENCE_ONLY[template]},
        {"role": "user", "content": f"Request: {request_text}\n\nDecision log:\n{body}\n\nWrite the explanation now."},
    ]


def cite_only_messages(request_text: str, prediction: str, probability: float, template: str = "t1") -> List[dict]:
    """Citation contract without the evidence, over the two citable fields."""
    _check_template(template)
    user = (
        f"Request: {request_text}\n\n"
        f"Log fields available for citation:\n"
        f"- prediction.class_index = {prediction} [prediction.class_index]\n"
        f"- prediction.probability = {probability:.2f} [prediction.probability]\n\n"
        f"Explain the prediction."
    )
    return [
        {"role": "system", "content": _SYSTEMS_CITE_ONLY[template]},
        {"role": "user", "content": user},
    ]

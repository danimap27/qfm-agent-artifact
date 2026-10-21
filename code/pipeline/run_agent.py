"""Run the agent pipeline end to end for the pilot conditions.

Requests to decision logs to verbalized explanations. Each request runs once
against the frozen checkpoint to produce a verifiable log, then every
condition verbalizes it. Decision logs can be prebuilt with
``code.pipeline.build_logs`` and passed through ``--logs``, which keeps the
remote runner free of the substrate and dataset dependencies. All language
model calls flow through the cached client, so a rerun with ``--replay``
reuses the exact responses. The verbalizer template (t1 to t3) varies only
the packaging of the answer, never the citation contract.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from code.evidence.manipulations import occlude_log, sabotage_log
from code.llm.ollama_client import OllamaClient
from code.pipeline.logs import build_logs, load_logs, write_logs
from code.verbalizer.prompts_2x2 import cite_only_messages, evidence_only_messages
from code.verbalizer.prompts import (
    free_messages,
    grounded_messages,
    prompt_hash,
    single_block_messages,
    template_floor_text,
)

LLM_CONDITIONS = ["grounded", "sabotage", "occlusion_attributions", "free", "single_block", "evidence_only", "cite_only"]


def build_messages(condition: str, item: dict, log: dict, task_ids: list[str], template: str = "t1") -> list[dict]:
    """Messages for one condition, with the matching evidence manipulation."""
    if condition == "grounded":
        return grounded_messages(item["text"], log, template=template)
    if condition == "sabotage":
        return grounded_messages(item["text"], sabotage_log(log), template=template)
    if condition == "occlusion_attributions":
        return grounded_messages(item["text"], occlude_log(log, ["attributions"]), template=template)
    if condition == "free":
        return free_messages(
            item["text"], log["prediction"]["label"], log["prediction"]["probability"], template=template
        )
    if condition == "single_block":
        return single_block_messages(item["text"], item["features"], task_ids, template=template)
    if condition == "evidence_only":
        return evidence_only_messages(item["text"], log, template=template)
    if condition == "cite_only":
        return cite_only_messages(
            item["text"], log["prediction"]["label"], log["prediction"]["probability"], template=template
        )
    raise ValueError(f"unknown llm condition {condition}")


def parse_single_block(text: str) -> dict:
    """Extract the claimed task, class, and explanation from the fixed format."""
    claimed_task = claimed_class = None
    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("task:"):
            claimed_task = line.split(":", 1)[1].strip()
        elif lowered.startswith("class:"):
            claimed_class = line.split(":", 1)[1].strip()
    marker = text.lower().find("explanation:")
    parsed_ok = marker >= 0
    explanation = text[marker + len("explanation:"):].strip() if parsed_ok else text.strip()
    return {"claimed_task": claimed_task, "claimed_class": claimed_class, "explanation": explanation, "parsed": parsed_ok}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="results/pilot/requests.json")
    parser.add_argument("--checkpoint", default="results/sweep_d4_b1900/model.npz")
    parser.add_argument("--out", default="results/pilot")
    parser.add_argument("--logs", default=None, help="prebuilt logs.jsonl; skips log construction locally")
    parser.add_argument("--template", default="t1", choices=["t1", "t2", "t3"])
    parser.add_argument("--llm-model", default="qwen3.5:4b")
    parser.add_argument("--llm-endpoint", default="http://localhost:11434/api/chat")
    parser.add_argument("--conditions", default=",".join(["template_floor", *LLM_CONDITIONS]))
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    request_file = json.loads(Path(args.requests).read_text())
    requests = request_file["requests"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.logs:
        logs_path = Path(args.logs)
        if not logs_path.exists():
            raise SystemExit(f"logs file not found: {logs_path}")
        logs = load_logs(logs_path)
        print(f"loaded {len(logs)} prebuilt logs from {logs_path}")
    else:
        logs = build_logs(requests, request_file.get("split_seed", 0), args.checkpoint)
        logs_path = write_logs(logs, out_dir / "logs.jsonl")
        print(f"wrote {logs_path} with {len(logs)} logs")

    task_ids = request_file.get("task_ids")
    if task_ids is None:
        from tasks.registry import SPECS

        task_ids = sorted(SPECS)

    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    client = OllamaClient(out_dir / "llm_cache.jsonl", model=args.llm_model, endpoint=args.llm_endpoint, replay=args.replay)
    records_path = out_dir / "explanations.jsonl"
    finished = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                finished.add((record["request_id"], record["condition"]))
        print(f"resuming, {len(finished)} records already present")
    with records_path.open("a") as handle:
        for item in requests:
            log = logs[item["id"]]
            for condition in conditions:
                if (item["id"], condition) in finished:
                    continue
                started = time.time()
                if condition == "template_floor":
                    response = template_floor_text(log)
                    parsed = {"explanation": response, "parsed": True}
                    llm_model = "none"
                    hash_value = None
                    tokens = {"eval_count": None, "prompt_eval_count": None}
                    repaired = False
                else:
                    messages = build_messages(condition, item, log, task_ids, args.template)
                    call = client.chat(messages)
                    response = call["response"]
                    parsed = (
                        parse_single_block(response)
                        if condition == "single_block"
                        else {"explanation": response.strip(), "parsed": True}
                    )
                    llm_model = client.model
                    hash_value = prompt_hash(messages)
                    tokens = {"eval_count": call.get("eval_count"), "prompt_eval_count": call.get("prompt_eval_count")}
                    repaired = bool(call.get("repaired", False))
                record = {
                    "request_id": item["id"],
                    "condition": condition,
                    "template": args.template,
                    "llm_model": llm_model,
                    "prompt_hash": hash_value,
                    "response": response,
                    "parsed": parsed,
                    "repaired": repaired,
                    "seconds": round(time.time() - started, 2),
                    "tokens": tokens,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(f"{item['id']:<14} {condition:<26} {record['seconds']:6.1f}s")

    print(f"cache hits {client.hits} misses {client.misses}, {client.seconds:.0f}s in the model")
    print(f"wrote {records_path}")


if __name__ == "__main__":
    main()

"""Build the request sets for the pipeline: real test samples turned into requests.

The pilot default covers six tasks with five samples each. The grid builds
larger sets by sampling rows per request seed. The features are the encoded
angles the substrate consumes, so every request is exactly executable by the
frozen checkpoint. The set is deterministic for a fixed sample seed. The file
records the split seed and the full candidate task list, so remote runners do
not need the task registry to rebuild prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tasks.registry import SPECS, get_tasks

PILOT_TASKS = ["iris", "wine", "wdbc", "thyroid", "parkinsons", "digits"]

REQUEST_TEMPLATES = {
    "iris": "Case {case}: routine botanical screening of a flower specimen from petal and sepal measurements.",
    "wine": "Case {case}: a winery submitted a wine sample for chemical analysis of its cultivar.",
    "wdbc": "Case {case}: a cytology sample of a breast mass needs a malignancy assessment.",
    "thyroid": "Case {case}: a thyroid function panel requires a diagnosis for this patient.",
    "parkinsons": "Case {case}: sustained vowel phonation recordings were submitted for Parkinson's screening.",
    "digits": "Case {case}: a small handwritten digit scan needs to be recognized.",
}

SAMPLES_PER_TASK = 5
SPLIT_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/pilot/requests.json")
    parser.add_argument("--tasks", default=",".join(PILOT_TASKS))
    parser.add_argument("--per-task", type=int, default=SAMPLES_PER_TASK)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="0 keeps the deterministic evenly spaced rows, positive values sample rows per seed",
    )
    args = parser.parse_args()

    names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    tasks = get_tasks(names, seed=SPLIT_SEED)
    requests = []
    for name in names:
        task = tasks[name]
        if args.sample_seed:
            rng = np.random.default_rng(args.sample_seed)
            rows = np.sort(rng.choice(task.X_test.shape[0], size=args.per_task, replace=False))
        else:
            rows = np.linspace(0, task.X_test.shape[0] - 1, args.per_task).astype(int)
        for position, row in enumerate(rows):
            requests.append(
                {
                    "id": f"{name}-{position + 1}",
                    "task": name,
                    "text": REQUEST_TEMPLATES[name].format(case=4100 + int(row)),
                    "features": [round(float(value), 6) for value in task.X_test[row]],
                    "gold_class": int(task.y_test[row]),
                    "n_classes": int(task.n_classes),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"split_seed": SPLIT_SEED, "task_ids": sorted(SPECS), "tasks": names, "requests": requests},
            indent=2,
        )
    )
    print(f"wrote {out} with {len(requests)} requests over {len(names)} tasks")


if __name__ == "__main__":
    main()

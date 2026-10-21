"""Build the expanded (v3) routing probe: twelve requests per task, 156 total.

Extends the v2 probe (four requests per task) with eight new phrasing
variants per task. The per-task order becomes [1, 2, 5, 6, 7, 8, 3, 4, 9,
10, 11, 12]: the first half (the training split of the classical baselines)
keeps the two original training items and gains four new ones, and the
second half (the shared evaluation split) keeps the two original evaluation
items and gains four new ones, so results on the original probe remain a
reproducible subset. Deterministic: the variants are fixed strings.

Usage: .venv/bin/python -m code.router.build_probe --out data/router_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NEW_VARIANTS: dict[str, list[str]] = {
    "iris": [
        "iris species identification from flower measurements",
        "Botanical classification: which iris species is this sample?",
        "Sort iris samples into species classes using their petal and sepal sizes.",
        "I collected petal and sepal sizes from several flowers, help me label the iris species.",
        "Determine the iris species class from four morphometric features of the flower.",
        "Classify the flower type, similar to iris species discrimination from sepal and petal data.",
        "iris: species label from petal length, petal width, sepal length, sepal width",
        "A researcher studying wild irises wants to assign each specimen to its species using simple flower dimension readings.",
    ],
    "wine": [
        "wine cultivar classification from chemistry",
        "Which of the three wine cultivars does this chemical sample belong to?",
        "Map wine chemical measurements to their cultivar of origin.",
        "I ran a chemical analysis on a wine bottle, can you tell me the grape cultivar?",
        "Identify the cultivar class of a wine sample from its laboratory chemistry features.",
        "Wine type prediction, similar to cultivar recognition from lab values.",
        "wine: cultivar from alcohol, malic acid, ash and related chemical columns",
        "A winery lab wants to verify the stated cultivar of a bottle by checking its chemical fingerprint.",
    ],
    "wdbc": [
        "breast tumor malignancy classification from cell measurements",
        "Is this breast mass malignant or benign based on its cell nuclei features?",
        "Assign a malignant or benign label to a breast tissue sample from cytology measurements.",
        "The doctor needs to know if the lump is cancerous, here are the cell nucleus numbers.",
        "Classify breast mass samples as malignant or benign from fine needle aspirate features.",
        "Tumor screening decision, similar to breast lesion classification from cytology.",
        "wdbc: diagnosis label from radius, texture, perimeter and smoothness of cell nuclei",
        "A pathology team wants an automated read of breast mass aspiration measurements into malignant or benign.",
    ],
    "thyroid": [
        "thyroid function classification from hormone panels",
        "Does this hormone panel indicate hyperthyroidism, hypothyroidism or normal function?",
        "Label thyroid lab panels as normal, underactive or overactive.",
        "I have blood test results for thyroid hormones, is the patient's thyroid overacting or underacting?",
        "Classify thyroid status from routine hormone assay values.",
        "Endocrine screening decision, similar to thyroid condition diagnosis from blood panels.",
        "thyroid: condition from TSH, T3 and T4 hormone readings",
        "A clinic wants to automatically triage thyroid function test results into disease categories.",
    ],
    "parkinsons": [
        "Parkinson screening from voice recordings",
        "Do these voice measurements show signs of Parkinson's disease?",
        "Detect early Parkinsonian patterns in sustained vowel recordings.",
        "My relative did a voice test, could these frequency and jitter numbers indicate Parkinson's?",
        "Screen for Parkinson's disease using biomedical voice signal characteristics.",
        "Neurodegenerative voice screening, similar to Parkinson detection from phonation data.",
        "parkinsons: healthy or affected from jitter, shimmer and harmonicity of sustained vowels",
        "A neurology study wants to flag Parkinsonian voice deterioration from sustained phonation samples.",
    ],
    "wine_quality": [
        "red wine quality score prediction from chemistry",
        "What sensory quality rating would this wine get based on its lab tests?",
        "Estimate the quality score of red wines from their physicochemical profile.",
        "Can you guess the quality score this red wine received from the taster panel, using the lab numbers?",
        "Predict the sensory quality label of red wine samples from analytical chemistry measurements.",
        "Wine grading, similar to sensory score estimation from acidity and sulphates.",
        "wine quality: score from fixed acidity, volatile acidity, citric acid and alcohol",
        "A cellar wants to rank its red wines by expected panel score using routine chemistry tests.",
    ],
    "sonar": [
        "sonar target discrimination: metal cylinder or rock",
        "Did this underwater echo come from a mine-like metal object or a rock?",
        "Classify sonar returns as metal cylinders or rocks from their spectral energy.",
        "We recorded sonar pings, can you say if the object was a metallic canister or a rocky bottom?",
        "Discriminate underwater sonar reflections from cylindrical metal targets versus rocks.",
        "Underwater object classification, similar to mine versus rock echo discrimination.",
        "sonar: target type from frequency band energies of the return",
        "A marine survey needs returns sorted into metal cylinder or rock classes from their chirp response.",
    ],
    "digits": [
        "handwritten digit recognition on tiny images",
        "Which digit is written in this low resolution handwriting sample?",
        "Read the digit from an eight by eight pixel grayscale scan.",
        "Here is a tiny picture of a handwritten number, what digit is it?",
        "Classify small handwritten digit images into their numeral classes.",
        "Character reading, similar to digit recognition in compact grayscale scans.",
        "digits: numeral from 64 pixel intensities of a handwritten sample",
        "An OCR prototype needs to turn small handwriting scans into the digit they show.",
    ],
    "mnist8": [
        "handwritten digit classification at scale",
        "Which numerals appear in this large batch of handwriting photos?",
        "Run digit recognition across a big collection of handwriting images.",
        "We scanned thousands of handwritten numbers, can you sort them by digit?",
        "Classify handwritten numerals from a broad corpus of digit photographs.",
        "Large scale character recognition, similar to digit sorting over image databases.",
        "mnist8: numeral labels for a mass of scanned handwritten digits",
        "A digitization project needs automated labels for a wide archive of handwritten digit photographs.",
    ],
    "fashion8": [
        "clothing category classification from photos",
        "Which garment category does this grayscale clothing photo show?",
        "Sort apparel images into categories like shoe, dress or coat.",
        "I have pictures of clothes, can you tell me which type of item each shows?",
        "Classify grayscale photographs of garments into fashion categories.",
        "Apparel sorting, similar to clothing category recognition in images.",
        "fashion8: item type from a grayscale product photo",
        "An online store wants to auto-tag clothing photos into garment categories like shirts or bags.",
    ],
    "periodic_sin": [
        "sign prediction for a periodic time series",
        "Will the next sample of this wave be above or below zero?",
        "Forecast the polarity of the next point in a thresholded sine sequence.",
        "We are tracking a sine wave, is the coming value positive or negative?",
        "Predict whether a periodic signal crosses zero at the next step.",
        "Signal polarity forecasting, similar to sign prediction in sinusoidal traces.",
        "periodic_sin: next step sign from the recent wave samples",
        "A monitoring tool needs the next reading of a periodic sensor classified as above or below the threshold.",
    ],
    "ionosphere": [
        "ionospheric radar return classification",
        "Is this radar echo from the ionosphere a good or bad return?",
        "Classify upper atmosphere radar returns as structured or noisy.",
        "Our radar picked up signals over the ionosphere, do they show a plasma structure or just noise?",
        "Decide whether ionospheric backscatter readings indicate electron density structures.",
        "Radar echo interpretation, similar to good versus bad ionosphere returns.",
        "ionosphere: good or bad echo from radar autocorrelation features",
        "A space weather station wants radar returns labeled as showing a structure or not.",
    ],
    "blood_transfusion": [
        "blood donor return prediction",
        "Will this donor come back to give blood again?",
        "Estimate repeat donation behavior from donor history records.",
        "We have a donor list, can you say who is likely to donate again?",
        "Predict the return of blood donors from their historical donation data.",
        "Donor retention scoring, similar to predicting repeat blood donations.",
        "blood_transfusion: returner or not from months since first and last donation",
        "A blood bank wants to prioritize outreach to donors likely to return for another donation.",
    ],
}

DESCRIPTION = (
    "Labeled request examples for the QFM-Agent routing probe. Twelve requests "
    "per task: the first half of each task's requests forms the classical-baseline "
    "training split, the second half is the shared evaluation split. Version 3 "
    "expands the v2 four-per-task probe and keeps its items inside their original "
    "halves as a reproducible subset."
)


def build(probe: dict) -> dict:
    """v2 probe dict to v3 probe dict, preserving per-task half membership."""
    by_task: dict[str, list[dict]] = {}
    for item in probe["requests"]:
        by_task.setdefault(item["task"], []).append(item)
    requests: list[dict] = []
    for task in sorted(by_task):
        originals = by_task[task]
        if len(originals) != 4:
            raise SystemExit(f"expected 4 requests for {task}, found {len(originals)}")
        variants = NEW_VARIANTS.get(task)
        if not variants or len(variants) != 8:
            raise SystemExit(f"expected 8 new variants for {task}")
        new_items = [{"id": f"{task}-{index + 5}", "task": task, "text": text} for index, text in enumerate(variants)]
        ordered = [originals[0], originals[1], *new_items[:4], originals[2], originals[3], *new_items[4:]]
        requests.extend(ordered)
    return {
        "version": 3,
        "created": "2026-09-28",
        "description": DESCRIPTION,
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/router_probe.json")
    parser.add_argument("--out", default="data/router_probe.json")
    args = parser.parse_args()
    source = json.loads(Path(args.probe).read_text())
    if source.get("version") != 2:
        raise SystemExit(f"expected a v2 probe, found version {source.get('version')}")
    built = build(source)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(built, indent=2) + "\n")
    counts: dict[str, int] = {}
    for item in built["requests"]:
        counts[item["task"]] = counts.get(item["task"], 0) + 1
    print(f"wrote {out_path}: {len(built['requests'])} requests over {len(counts)} tasks")
    for task, count in sorted(counts.items()):
        print(f"  {task:<20} {count}")


if __name__ == "__main__":
    main()

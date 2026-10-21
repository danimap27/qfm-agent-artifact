# Annotation protocol for the citation auditor

Purpose: measure how well the automatic citation auditor agrees with people, so
the unsupported rate can be reported with a human-validated error bound.

## Material

- `results/post_freeze/annotation/sheet_blind.csv`: 300 sentences, each with the
  evidence the verbalizer was shown. The auditor's verdict is not in the sheet.
- `results/post_freeze/annotation/key.jsonl`: the auditor's verdict and the
  provenance of each item. Annotators must not open it.
- The sample holds 75 sentences per citing condition (grounded, sabotage,
  occlusion, contract-only), balanced over the three verbalizers and over the
  two auditor verdicts.

## Task

Two annotators label every row independently in the column
`supported (1 yes / 0 no)` and may leave a comment in `notes`.

Label 1 when every factual statement in the sentence is backed by the evidence
shown in the same row, read at the rounding of the sentence (0.94 supports
0.9418, 94% supports 0.9418). Label 0 when any statement is absent from the
evidence, contradicts it, or names a field that does not hold the stated value.
The bracketed citation is a hint, not a requirement: judge the statement
against the whole evidence block, and do not penalize format (for example two
fields inside one bracket). Statements about the model's general behaviour that
the evidence cannot confirm are labelled 0.

Expected effort: about 20 seconds per row, under two hours per annotator.

## Agreement

    python scripts/annotation_package.py kappa annotator_a.csv annotator_b.csv

The script prints Cohen's kappa between the two annotators and between each
annotator and the auditor, with bootstrap 95 percent intervals over sentences.
Report the three values in the Threats to Validity section, and treat
disagreements between the auditor and both annotators as auditor errors.

# F1 sweep: substrate configuration selection (2026-09-23)

Goal: choose the pretrained substrate configuration for the agent phases by
measuring how much of the gap to a classical reference is budget and depth.

Grid: depth in {3, 4} times total SPSA steps in {880, 1900} (12x80 and
20x100), seed 0, the eleven-task pretraining mix with the balanced per-task
loss. Reference: multinomial logistic regression on the same eight features.
Runs are deterministic and take between three and eight minutes on the
homelab CPU.

| task | d3/880 | d3/1900 | d4/880 | d4/1900 | classical |
|---|---|---|---|---|---|
| iris | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wine | 0.778 | 0.806 | 0.917 | 0.917 | 0.972 |
| wdbc | 0.789 | 0.833 | 0.798 | 0.851 | 0.982 |
| thyroid | 0.923 | 0.923 | 0.927 | 0.937 | 0.933 |
| parkinsons | 0.744 | 0.744 | 0.692 | 0.744 | 0.821 |
| wine_quality | 0.516 | 0.503 | 0.550 | 0.575 | 0.584 |
| sonar | 0.738 | 0.714 | 0.738 | 0.714 | 0.786 |
| digits | 0.550 | 0.600 | 0.669 | 0.706 | 0.825 |
| mnist8 | 0.537 | 0.522 | 0.588 | 0.647 | 0.715 |
| fashion8 | 0.552 | 0.595 | 0.608 | 0.615 | 0.733 |
| periodic_sin | 0.800 | 0.842 | 0.758 | 0.875 | 0.967 |
| mean | 0.721 | 0.735 | 0.749 | 0.780 | 0.847 |

Decision: depth 4 with twenty outer rounds of 100 SPSA steps (1900 steps)
becomes the substrate configuration for the agent phases
(`configs/f1_reference.yaml`). Depth contributed more than budget at this
margin and the two combined. The trunk cross-entropy was still declining when
the budget ran out (0.619), so longer budgets remain a known direction. The
mean gap to the classical reference on the same features narrows from 0.126
to 0.067, and two tasks (iris, thyroid) match the reference.

Notes: the per-task test sets are small (30 to 600 samples), so single-task
swings of about five points are within noise. A seed replication and a longer
budget are reported below. Held-out tasks are excluded from the mix by design.

## Transfer with the reference model

Re-running the held-out protocol with the reference model: ionosphere reaches
0.873 with the probe and the identity adapter is kept, and blood transfusion
moves from a probe of 0.353 to 0.767 with the adapter, which recovers a task
whose frozen features are misaligned for a linear head. The selection rule
(best validation round, with the probe as an explicit candidate) prevents
degradation on either task.

## Seed replication and a longer budget (2026-09-24)

The reference configuration was replicated with seeds 1, 2, and 3. The mean
test accuracy over the eleven tasks is 0.780, 0.739, 0.780, and 0.759, so the
four-seed mean is 0.764 with a standard deviation of 0.017 and seed 0 is the
optimistic extreme. Per-task standard deviations range from 0.004 for thyroid
to 0.097 for wine, and wine quality stays weak in every seed (0.487 to 0.575),
consistent with the feature ceiling seen in the grid.

A longer run (thirty outer rounds, 2900 steps, seed 0) reaches a mean of 0.789
against 0.780 at 1900 steps, with gains concentrated in wine (0.944), periodic
sine (0.900), and digits (0.731). Trunk cross-entropy was still declining at
the end (0.598), at roughly a tenth of the earlier rate, so the 1900-step
budget sits close to its point of diminishing returns and the reference
configuration stays as selected.

The held-out protocol was re-run on every model. Ionosphere keeps the identity
adapter in three of five runs and improves by 7 and 21 points in the two runs
where the probe was weak. Blood transfusion is rescued by the adapter in four
of five runs, where probes of 0.34 to 0.40 rise to about 0.77, and keeps the
identity in the fifth. The selection rule adapts only when the frozen features
need it.

"""Figure: wrong-verdict rate of the counterfactual audit against the shot budget."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "results/post_freeze/device_noise_cf.json").read_text())

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.6, 2.5))
for label, series, style in (
    ("Ideal sampling", data["ideal"]["wrong_verdict_rate"], "o-"),
    ("Heron r2 noise model", data["device_noise"]["wrong_verdict_rate"], "s-"),
    ("Pairwise Gaussian approximation", data["analytic_shot_noise"], "^--"),
):
    shots = sorted(int(k) for k in series)
    ax.plot(shots, [series[str(s)] for s in shots], style, label=label, markersize=3.5, linewidth=0.9)
ax.set_xscale("log")
ax.set_xlabel("Shots per expectation estimate")
ax.set_ylabel("Wrong-verdict rate")
ax.set_ylim(0, 0.55)
ax.grid(linewidth=0.3, alpha=0.5)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout()
out = ROOT / "paper/figures/fig11_device_noise.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

"""Collect run metrics into tables laid out like the paper, next to the paper's values.

Reads results/<data>/**/metrics.json (written by scripts/run_queue.py) and the
paper's numbers from scripts/paper_reference.json. Writes results/comparison.md
for v4 and results/comparison_v12.md for v12.

Each cell shows "ours (paper)". Ours is mean ± std over the seeds that have
finished, using the population std (numpy default), as in the paper. A cell is
marked † when our mean lies outside the paper's mean ± 2 std, and [n=k] when
fewer than 3 seeds are done.

Usage: python scripts/collect_results.py [--data v4|v12]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GROUPS = ["Li-ion", "Zn-ion", "Na-ion", "CALB"]
SEEDS = [42, 2021, 2024]


def load(results):
    runs = {}
    for f in results.glob("*/*/*/*/metrics.json"):
        table, model, group, seed = f.parent.relative_to(results).parts
        runs[(table, model, group, int(seed))] = json.loads(f.read_text())
    return runs


def parse_paper(text):
    try:
        mean, std = text.split("±")
        return float(mean), float(std)
    except (AttributeError, ValueError):
        return None


def cell(runs, key, metric, paper):
    """key = (table, model, group); returns 'ours (paper)'."""
    vals = [runs[key + (s,)][metric] for s in SEEDS
            if key + (s,) in runs and metric in runs[key + (s,)] and runs[key + (s,)][metric] >= 0]
    p = parse_paper(paper)
    paper_txt = paper if paper else "?"
    if not vals:
        return f"· ({paper_txt})"
    mean, std = float(np.mean(vals)), float(np.std(vals))
    ours = f"{mean:.3f}±{std:.3f}"
    if p and abs(mean - p[0]) > 2 * p[1]:
        ours += "†"
    if len(vals) < len(SEEDS):
        ours += f" [n={len(vals)}]"
    return f"{ours} ({paper_txt})"


def main_table(runs, ref):
    head = "| Method | " + " | ".join(f"{g} MAPE | {g} 15%-Acc" for g in GROUPS) + " |"
    lines = [head, "|" + "---|" * (1 + 2 * len(GROUPS))]
    for model, paper in ref["main"].items():
        row = [model]
        for g in GROUPS:
            for metric, pm in (("test_MAPE", "MAPE"), ("test_acc15", "Acc")):
                row.append(cell(runs, ("main", model, g), metric, paper[g][pm]))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def ablation_table(runs, ref):
    lines = ["| Model | Variant | " + " | ".join(f"{g} MAPE | {g} 15%-Acc" for g in GROUPS) + " |",
             "|" + "---|" * (2 + 2 * len(GROUPS))]
    for model, variants in ref["ablation"].items():
        for variant, paper in variants.items():
            source = {"Full": ("main", model), "MLP": ("main", "MLP"), "Transformer": ("main", "Transformer"),
                      "w/o Intra": ("ablation", f"{model}-wo-intra"),
                      "w/o Inter": ("ablation", f"{model}-wo-inter")}[variant]
            row = [model, variant]
            for g in GROUPS:
                for metric, pm in (("test_MAPE", "MAPE"), ("test_acc15", "Acc")):
                    row.append(cell(runs, source + (g,), metric, paper[g][pm]))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def seen_unseen_table(runs, ref):
    lines = ["| Data | Model | MAPE seen | MAPE unseen | 15%-Acc seen | 15%-Acc unseen |", "|---|---|---|---|---|---|"]
    for g, models in ref["seen_unseen"].items():
        for model, paper in models.items():
            key = ("main", model, g)
            lines.append(f"| {g} | {model} | "
                         + " | ".join([cell(runs, key, "test_seen_MAPE", paper["MAPE"]["Seen"]),
                                       cell(runs, key, "test_unseen_MAPE", paper["MAPE"]["Unseen"]),
                                       cell(runs, key, "test_seen_acc15", paper["Acc"]["Seen"]),
                                       cell(runs, key, "test_unseen_acc15", paper["Acc"]["Unseen"])]) + " |")
    return lines


def transfer_table(runs, ref):
    lines = ["| Target | Metric | Paper baseline | Frozen | FT | DA |", "|---|---|---|---|---|---|"]
    for g, metrics in ref["transfer"].items():
        for pm, metric in (("MAPE", "test_MAPE"), ("Acc", "test_acc15")):
            paper = metrics[pm]
            row = [g, "MAPE" if pm == "MAPE" else "15%-Acc", paper["Baseline"]]
            row += [cell(runs, ("transfer", m, g), metric, paper[m]) for m in ("Frozen", "FT", "DA")]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="v4", choices=["v4", "v12"])
    args = p.parse_args()
    ref = json.loads((REPO / "scripts" / "paper_reference.json").read_text())
    runs = load(REPO / "results" / args.data)
    out = REPO / "results" / ("comparison.md" if args.data == "v4" else f"comparison_{args.data}.md")

    text = [f"# BatteryLife replication: results on data {args.data}", "",
            f"{len(runs)} finished runs. Cells: ours (paper). Ours = mean ± std over finished seeds "
            "(42, 2021, 2024), population std. † = outside paper mean ± 2 std. "
            "[n=k] = only k seeds finished. · = not run yet.", "",
            "## Table 3: main benchmark", ""] + main_table(runs, ref) + [
            "", "## Table 4: ablation", ""] + ablation_table(runs, ref) + [
            "", "## Table 5: seen vs unseen aging conditions", ""] + seen_unseen_table(runs, ref) + [
            "", "## Table 6: cross-domain transfer (CPMLP pretrained on Li-ion)", ""] + transfer_table(runs, ref)
    out.write_text("\n".join(text) + "\n")
    print(f"wrote {out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.plot_results import (
    plot_comparative_pareto,
    plot_metric_comparison,
    plot_training_convergence,
)
from utils.stats_agg import (
    METRIC_META,
    build_ablation_contributions,
    build_comparative_analysis,
    export_latex_tables,
    export_markdown_report,
    export_summary_csv,
    load_experiment_records,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze HRL-VGAE experiments and generate publication report")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--logs-root", type=str, default="logs")
    parser.add_argument("--out", type=str, default="report")
    parser.add_argument("--formats", type=str, nargs="+", default=["png", "pdf"])
    parser.add_argument("--pareto-x", type=str, default="cost", choices=["cost", "cost_per_acc"])
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    logs_root = Path(args.logs_root) if args.logs_root else None
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = load_experiment_records(results_root, logs_root=logs_root)
    if not records:
        print(f"[WARN] No experiment records found under {results_root}")
        return
    comparative = build_comparative_analysis(records)
    ablation_rows = build_ablation_contributions(comparative, reference="HRL-VGAE")

    (out_dir / "comparative_stats.json").write_text(json.dumps({
        "comparative": comparative,
        "ablation_contributions": ablation_rows,
    }, indent=2, allow_nan=False))
    export_summary_csv(comparative, out_dir / "comparative_summary.csv")
    export_latex_tables(comparative, out_dir / "tables.tex")
    export_markdown_report(comparative, ablation_rows, records, out_dir / "research_summary.md")

    for metric_key, meta in METRIC_META.items():
        arrow = "↑" if meta["higher_better"] else "↓"
        plot_metric_comparison(comparative, metric_key, f"{meta['label']} ({arrow})", str(fig_dir), formats=args.formats)

    by_inst = {}
    for r in records:
        by_inst.setdefault(r["base_instance"], []).append(r)
    for inst, inst_records in sorted(by_inst.items()):
        inst_fig_dir = fig_dir / inst
        mo_path = results_root / f"{inst}_mo_indicators.json"
        plot_comparative_pareto(
            inst, inst_records, str(inst_fig_dir),
            x_mode=args.pareto_x,
            mo_json_path=str(mo_path) if mo_path.is_file() else "",
            formats=args.formats,
        )
        plot_training_convergence(inst, inst_records, str(inst_fig_dir), window=args.window, formats=args.formats)

    print(f"Report generated -> {out_dir.resolve()} ({len(records)} runs analyzed)")


if __name__ == "__main__":
    main()
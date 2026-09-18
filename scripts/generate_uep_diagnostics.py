#!/usr/bin/env python3
"""Generate the normalized UEP diagnostic-results CSV from summary files."""

import argparse
import csv
import json
from pathlib import Path


RESULT_ROOTS = [
    "out_fixed_budget_uep_matched_validation",
    "out_fixed_budget_uep_low07_matched_validation",
    "out_fixed_budget_uep_low07_best_top1_matched_validation",
    "out_fixed_budget_uep_low08_matched_validation",
    "out_fixed_budget_uep_low08_best_top1_matched_validation",
    "out_fixed_budget_uep_aggressive_best_top1_matched_validation",
    "out_fixed_budget_mild_r2plus1d_v3_matched_validation",
    "out_fixed_budget_moderate_r2plus1d_v3_matched_validation",
    "out_fixed_budget_aggressive_r2plus1d_v3_matched_validation",
    "out_continuous_uep_alpha02_primary_matched_validation",
    "out_continuous_uep_r2plus1d_matched_validation",
    "out_staged_varnorm_alpha02_primary_matched_validation",
    "out_staged_varnorm_alpha02_r2plus1d_matched_validation",
]

FIELDNAMES = [
    "result_dir",
    "evaluator",
    "method",
    "mode",
    "psnr_db",
    "ssim",
    "ms_ssim_3scale",
    "top1_percent",
    "top5_percent",
    "clean_top1_percent",
]


def classify_evaluator(path: Path) -> str:
    return "r2plus1d" if "r2plus1d" in str(path).lower() else "tsn"


def classify_method(path: Path) -> str:
    text = str(path).lower()
    if "continuous" in text:
        return "continuous_uep"
    if "staged_varnorm" in text:
        return "staged_varnorm_uep"
    return "fixed_budget_uep"


def collect_rows(repo_root: Path) -> list[dict]:
    rows = []
    summary_files = []

    for relative_root in RESULT_ROOTS:
        result_root = repo_root / relative_root
        if not result_root.is_dir():
            raise FileNotFoundError(f"Missing result directory: {result_root}")

        root_summaries = list(result_root.rglob("summary.json"))
        if not root_summaries:
            raise FileNotFoundError(
                f"No summary.json found under: {result_root}"
            )
        summary_files.extend(root_summaries)

    for summary_path in sorted(summary_files):
            with summary_path.open(encoding="utf-8") as stream:
                document = json.load(stream)

            summaries = document.get("summaries")
            if not isinstance(summaries, dict):
                raise ValueError(
                    f"Missing or invalid 'summaries' object: {summary_path}"
                )

            relative_result_dir = summary_path.parent.relative_to(repo_root)

            for mode, metrics in summaries.items():
                rows.append(
                    {
                        "result_dir": str(relative_result_dir),
                        "evaluator": classify_evaluator(summary_path),
                        "method": classify_method(summary_path),
                        "mode": mode,
                        "psnr_db": metrics.get("psnr_db"),
                        "ssim": metrics.get("ssim"),
                        "ms_ssim_3scale": metrics.get(
                            "ms_ssim_3scale"
                        ),
                        "top1_percent": metrics.get("top1_percent"),
                        "top5_percent": metrics.get("top5_percent"),
                        "clean_top1_percent": metrics.get(
                            "clean_top1_percent"
                        ),
                    }
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else repo_root / "canonical_uep_diagnostics.csv"
    )

    rows = collect_rows(repo_root)

    if len(rows) != 52:
        raise RuntimeError(
            f"Expected 52 diagnostic rows, found {len(rows)}"
        )

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate canonical manuscript-facing results from frozen JSON summaries."""

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "split",
    "family",
    "mode",
    "evaluator",
    "training_seed",
    "channel",
    "c",
    "snr_db",
    "nominal_cbr",
    "actual_cbr",
    "channel_seed",
    "clips",
    "psnr_db",
    "ssim",
    "ms_ssim_3scale",
    "top1_percent",
    "top5_percent",
    "clean_top1_percent",
    "clean_top5_percent",
    "official_test_used",
    "source_summary",
]


def parse_metadata(path: Path) -> dict:
    text = str(path)
    lower = text.lower()

    channel_match = re.search(r"_(AWGN|Rayleigh)_", text)
    c_match = re.search(r"_c([48])_", text)
    snr_match = re.search(r"_snr(-?\d+(?:\.\d+)?)", text)

    if not channel_match or not c_match or not snr_match:
        raise ValueError(f"Cannot parse channel/c/SNR from {path}")

    c_value = int(c_match.group(1))

    if "seed43" in lower:
        training_seed = 43
    elif "seed44" in lower:
        training_seed = 44
    else:
        training_seed = 42

    evaluator = None
    if "r2plus1d" in lower:
        evaluator = "r2plus1d"
    elif "tsn" in lower:
        evaluator = "tsn"

    return {
        "channel": channel_match.group(1),
        "c": c_value,
        "snr_db": float(snr_match.group(1)),
        "training_seed": training_seed,
        "evaluator": evaluator,
        "nominal_cbr": "1/6" if c_value == 8 else "1/12",
    }


def classify_family(path: Path, c_value: int) -> str:
    lower = str(path).lower()

    if "/topk/" in lower or "topk50" in lower:
        return "exact_topk"

    if "correction_finetune" in lower:
        return "fine_tune_only_c8"

    if (
        "out_no_selector_" in lower
        and "_primary_locked_validation" in lower
    ):
        return "fine_tune_only_c8"

    if "finetuned_native_c4" in lower:
        return "fine_tuned_native_c4"

    return f"reconstruction_only_c{c_value}"


def make_row(
    *,
    path: Path,
    repo_root: Path,
    document: dict,
    metrics: dict,
    evaluator: str,
    family: str,
    mode: str,
    actual_cbr: str,
) -> dict:
    meta = parse_metadata(path)

    return {
        "split": document.get(
            "split",
            "test" if document.get("official_test_used") else "validation",
        ),
        "family": family,
        "mode": mode,
        "evaluator": evaluator,
        "training_seed": meta["training_seed"],
        "channel": meta["channel"],
        "c": meta["c"],
        "snr_db": meta["snr_db"],
        "nominal_cbr": meta["nominal_cbr"],
        "actual_cbr": actual_cbr,
        "channel_seed": document.get(
            "channel_seed",
            metrics.get("channel_seed", 1042),
        ),
        "clips": metrics.get("clips", document.get("clips")),
        "psnr_db": metrics.get("psnr_db"),
        "ssim": metrics.get("ssim"),
        "ms_ssim_3scale": metrics.get("ms_ssim_3scale"),
        "top1_percent": metrics.get("top1_percent"),
        "top5_percent": metrics.get("top5_percent"),
        "clean_top1_percent": metrics.get("clean_top1_percent"),
        "clean_top5_percent": metrics.get("clean_top5_percent"),
        "official_test_used": document.get(
            "official_test_used",
            document.get("split") == "test",
        ),
        "source_summary": str(path.relative_to(repo_root)),
    }


def expand_topk(path: Path, repo_root: Path, document: dict) -> list[dict]:
    meta = parse_metadata(path)
    evaluator = meta["evaluator"]

    if evaluator is None:
        raise ValueError(f"Cannot infer evaluator from {path}")

    summaries = document.get("summaries")
    if not isinstance(summaries, dict):
        raise ValueError(f"Missing top-k summaries in {path}")

    expected_modes = {
        "learned_topk",
        "random_topk",
        "uniform_topk",
    }
    if set(summaries) != expected_modes:
        raise ValueError(
            f"Unexpected top-k modes in {path}: {sorted(summaries)}"
        )

    return [
        make_row(
            path=path,
            repo_root=repo_root,
            document=document,
            metrics=metrics,
            evaluator=evaluator,
            family="exact_topk",
            mode=mode,
            actual_cbr="1/12",
        )
        for mode, metrics in summaries.items()
    ]


def expand_combined_test(
    path: Path,
    repo_root: Path,
    document: dict,
) -> list[dict]:
    meta = parse_metadata(path)
    family = classify_family(path, meta["c"])
    rows = []

    for evaluator, prefix in (
        ("tsn", "tsn"),
        ("r2plus1d", "r2plus1d"),
    ):
        metrics = {
            "clips": document.get("clips"),
            "psnr_db": document.get("psnr_db"),
            "ssim": document.get("ssim"),
            "ms_ssim_3scale": document.get("ms_ssim_3scale"),
            "top1_percent": document.get(
                f"{prefix}_reconstructed_top1_percent"
            ),
            "top5_percent": document.get(
                f"{prefix}_reconstructed_top5_percent"
            ),
            "clean_top1_percent": document.get(
                f"{prefix}_clean_top1_percent"
            ),
            "clean_top5_percent": document.get(
                f"{prefix}_clean_top5_percent"
            ),
        }

        rows.append(
            make_row(
                path=path,
                repo_root=repo_root,
                document=document,
                metrics=metrics,
                evaluator=evaluator,
                family=family,
                mode="no_selection",
                actual_cbr=meta["nominal_cbr"],
            )
        )

    return rows


def expand_validation(
    path: Path,
    repo_root: Path,
    document: dict,
) -> list[dict]:
    meta = parse_metadata(path)

    if meta["evaluator"] is None:
        raise ValueError(f"Cannot infer evaluator from {path}")

    family = classify_family(path, meta["c"])

    metrics = {
        "clips": document.get("clips"),
        "psnr_db": document.get("psnr_db"),
        "ssim": document.get("ssim"),
        "ms_ssim_3scale": document.get("ms_ssim_3scale"),
        "top1_percent": document.get("reconstructed_top1_percent"),
        "top5_percent": document.get("reconstructed_top5_percent"),
        "clean_top1_percent": document.get("clean_top1_percent"),
        "clean_top5_percent": document.get("clean_top5_percent"),
    }

    return [
        make_row(
            path=path,
            repo_root=repo_root,
            document=document,
            metrics=metrics,
            evaluator=meta["evaluator"],
            family=family,
            mode="no_selection",
            actual_cbr=meta["nominal_cbr"],
        )
    ]


def collect_rows(repo_root: Path, source_file: Path) -> list[dict]:
    source_paths = [
        repo_root / line.strip()
        for line in source_file.read_text().splitlines()
        if line.strip()
    ]

    if len(source_paths) != 64:
        raise RuntimeError(
            f"Expected 64 source summaries, found {len(source_paths)}"
        )

    rows = []

    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)

        if "summaries" in document:
            rows.extend(expand_topk(path, repo_root, document))
        elif document.get("split") == "test":
            rows.extend(expand_combined_test(path, repo_root, document))
        else:
            rows.extend(expand_validation(path, repo_root, document))

    return rows


def validate_rows(rows: list[dict]) -> None:
    if len(rows) != 109:
        raise RuntimeError(f"Expected 109 rows, found {len(rows)}")

    required = [
        "split",
        "family",
        "mode",
        "evaluator",
        "channel",
        "c",
        "snr_db",
        "clips",
        "psnr_db",
        "ssim",
        "ms_ssim_3scale",
        "top1_percent",
        "top5_percent",
        "clean_top1_percent",
        "clean_top5_percent",
        "source_summary",
    ]

    for index, row in enumerate(rows, start=1):
        missing = [
            field
            for field in required
            if row.get(field) is None or row.get(field) == ""
        ]
        if missing:
            raise RuntimeError(
                f"Row {index} is missing {missing}: "
                f"{row['source_summary']}"
            )

    correction_rows = [
        row
        for row in rows
        if "correction_finetune" in row["source_summary"]
    ]
    if len(correction_rows) != 2:
        raise RuntimeError(
            f"Expected 2 corrected fine-tune rows, "
            f"found {len(correction_rows)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_file = (
        args.sources.resolve()
        if args.sources
        else repo_root / "canonical_result_sources.txt"
    )
    output_file = (
        args.output.resolve()
        if args.output
        else repo_root / "canonical_results.csv"
    )

    rows = collect_rows(repo_root, source_file)
    validate_rows(rows)

    rows.sort(
        key=lambda row: (
            row["split"],
            row["family"],
            row["training_seed"],
            row["evaluator"],
            row["channel"],
            row["c"],
            row["snr_db"],
            row["mode"],
        )
    )

    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    validation_count = sum(
        row["split"] == "validation" for row in rows
    )
    test_count = sum(row["split"] == "test" for row in rows)

    print(f"Wrote {len(rows)} rows to {output_file}")
    print(f"Validation rows: {validation_count}")
    print(f"Official-test rows: {test_count}")


if __name__ == "__main__":
    main()

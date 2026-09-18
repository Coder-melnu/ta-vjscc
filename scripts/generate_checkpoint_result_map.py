#!/usr/bin/env python3
"""Map every canonical result row to its model and evaluator checkpoints."""

import argparse
import csv
from pathlib import Path


FIELDS = [
    "result_id",
    "split",
    "family",
    "mode",
    "evaluator",
    "training_seed",
    "channel",
    "c",
    "snr_db",
    "source_summary",
    "model_checkpoint",
    "model_checkpoint_sha256",
    "evaluator_checkpoint",
    "evaluator_checkpoint_sha256",
    "evaluation_script",
]


def read_hashes(path: Path) -> dict[str, str]:
    hashes = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        digest, filename = line.split(maxsplit=1)
        hashes[filename.strip()] = digest

    return hashes


def snr_label(value: str) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def select_model_checkpoint(
    row: dict,
    model_paths: list[str],
) -> str:
    family = row["family"]
    seed = int(row["training_seed"])
    channel = row["channel"]
    c_value = row["c"]
    snr = snr_label(row["snr_db"])

    if family == "exact_topk":
        candidates = [
            path
            for path in model_paths
            if "out_topk50_selector_trained" in path
        ]

        if seed == 42:
            candidates = [
                path for path in candidates
                if "_seed43/" not in path
                and "_seed44/" not in path
            ]
        else:
            candidates = [
                path for path in candidates
                if f"_seed{seed}/" in path
            ]

    elif family == "fine_tune_only_c8":
        candidates = [
            path
            for path in model_paths
            if "out_videojscc_no_selector_c8_snr13/" in path
            and path.endswith("/best_joint_loss.pt")
        ]

    elif family == "fine_tuned_native_c4":
        candidates = [
            path
            for path in model_paths
            if (
                f"out_finetuned_native_c4_snr13_seed{seed}_"
                in path
                and path.endswith("/best_joint_loss.pt")
            )
        ]

    elif family in {
        "reconstruction_only_c4",
        "reconstruction_only_c8",
    }:
        token = f"VideoJSCC_{channel}_c{c_value}_snr{snr}_"
        candidates = [
            path
            for path in model_paths
            if token in path
            and path.endswith("/checkpoints/best.pt")
        ]

    else:
        raise ValueError(f"Unknown family: {family}")

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one checkpoint for "
            f"{family}, seed={seed}, channel={channel}, "
            f"c={c_value}, snr={snr}; found {candidates}"
        )

    return candidates[0]


def select_evaluator_checkpoint(
    evaluator: str,
    all_paths: list[str],
) -> str:
    if evaluator == "tsn":
        suffix = "tsn_ucf101_head_locked_split_best.pt"
    elif evaluator == "r2plus1d":
        suffix = "r2plus1d_ucf101_layer4_locked_split_best.pt"
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

    candidates = [
        path for path in all_paths if path.endswith(suffix)
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one evaluator checkpoint for "
            f"{evaluator}; found {candidates}"
        )

    return candidates[0]


def select_evaluation_script(row: dict) -> str:
    if row["split"] == "test":
        if (
            row["family"] == "fine_tune_only_c8"
            and "correction_finetune" in row["source_summary"]
        ):
            return "eval_locked_controls.py"

        return "run_official_test_sweep.py"

    if row["family"] == "exact_topk":
        if row["evaluator"] == "tsn":
            return (
                "downstream/action_recognition/"
                "eval_topk_selector_matched_tsn.py"
            )

        return (
            "downstream/action_recognition/"
            "eval_topk_selector_matched_r2plus1d.py"
        )

    if row["evaluator"] == "tsn":
        return (
            "downstream/action_recognition/"
            "eval_videojscc_tsn_clean.py"
        )

    return (
        "downstream/action_recognition/"
        "eval_videojscc_r2plus1d_clean.py"
    )


def make_result_id(row: dict) -> str:
    snr = snr_label(row["snr_db"])

    return "_".join(
        [
            row["split"],
            row["family"],
            row["mode"],
            row["evaluator"],
            f"seed{row['training_seed']}",
            row["channel"],
            f"c{row['c']}",
            f"snr{snr}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--checksums", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_path = (
        args.results.resolve()
        if args.results
        else repo_root / "canonical_results.csv"
    )
    checksums_path = (
        args.checksums.resolve()
        if args.checksums
        else repo_root / "canonical_checkpoint_SHA256SUMS.txt"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else repo_root / "checkpoint_result_map.csv"
    )

    hashes = read_hashes(checksums_path)
    all_paths = sorted(hashes)
    model_paths = [
        path
        for path in all_paths
        if not path.startswith(
            "downstream/action_recognition/weights/"
        )
    ]

    with results_path.open(newline="", encoding="utf-8") as stream:
        result_rows = list(csv.DictReader(stream))

    output_rows = []

    for row in result_rows:
        model_checkpoint = select_model_checkpoint(
            row,
            model_paths,
        )
        evaluator_checkpoint = select_evaluator_checkpoint(
            row["evaluator"],
            all_paths,
        )

        output_rows.append(
            {
                "result_id": make_result_id(row),
                "split": row["split"],
                "family": row["family"],
                "mode": row["mode"],
                "evaluator": row["evaluator"],
                "training_seed": row["training_seed"],
                "channel": row["channel"],
                "c": row["c"],
                "snr_db": row["snr_db"],
                "source_summary": row["source_summary"],
                "model_checkpoint": model_checkpoint,
                "model_checkpoint_sha256": hashes[
                    model_checkpoint
                ],
                "evaluator_checkpoint": evaluator_checkpoint,
                "evaluator_checkpoint_sha256": hashes[
                    evaluator_checkpoint
                ],
                "evaluation_script": select_evaluation_script(row),
            }
        )

    if len(output_rows) != 109:
        raise RuntimeError(
            f"Expected 109 mapped rows, found {len(output_rows)}"
        )

    result_ids = [row["result_id"] for row in output_rows]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("Duplicate result_id values detected")

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

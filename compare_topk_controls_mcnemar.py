"""Exact paired McNemar tests for learned top-k versus locked controls."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_seed_path(value):
    try:
        seed_text, path = value.split("=", 1)
        return int(seed_text), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected SEED=CSV_PATH") from exc


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c4_csv", required=True, type=Path)
    p.add_argument("--no_selector_csv", type=Path)
    p.add_argument("--tsn", action="append", type=parse_seed_path, required=True)
    p.add_argument("--r2plus1d", action="append", type=parse_seed_path, required=True)
    p.add_argument("--out", type=Path, default=Path("topk_vs_controls_mcnemar.json"))
    return p.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1128:
        raise RuntimeError(f"{path}: expected 1128 rows, found {len(rows)}")
    required = {"clip_id", "label"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{path}: missing clip_id or label")
    return rows


def aligned(left, right, left_path, right_path):
    for index, (a, b) in enumerate(zip(left, right)):
        if a["clip_id"] != b["clip_id"] or a["label"] != b["label"]:
            raise RuntimeError(
                f"Row {index} is not aligned between {left_path} and {right_path}"
            )


def exact_mcnemar(left, right, left_field, right_field):
    left_win = right_win = 0
    for a, b in zip(left, right):
        x = int(a[left_field])
        y = int(b[right_field])
        left_win += int(x == 1 and y == 0)
        right_win += int(x == 0 and y == 1)
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(left_win, right_win)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_correct_right_wrong": left_win,
        "left_wrong_right_correct": right_win,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def compare(seed, evaluator, learned_path, learned, control_name, control_path, control):
    aligned(learned, control, learned_path, control_path)
    control_prefix = "tsn" if evaluator == "TSN" else "r2plus1d"
    result = {
        "seed": seed,
        "evaluator": evaluator,
        "left": "learned_topk",
        "right": control_name,
        "learned_csv": str(learned_path),
        "control_csv": str(control_path),
    }
    for k in (1, 5):
        test = exact_mcnemar(
            learned,
            control,
            f"reconstructed_top{k}_correct",
            f"{control_prefix}_reconstructed_top{k}_correct",
        )
        result[f"top{k}"] = test
    return result


def main():
    args = parser()
    c4 = read_rows(args.c4_csv)
    no_selector = read_rows(args.no_selector_csv) if args.no_selector_csv else None
    learned_sets = {
        "TSN": dict(args.tsn),
        "R(2+1)D": dict(args.r2plus1d),
    }
    expected_seeds = {42, 43, 44}
    for evaluator, mapping in learned_sets.items():
        if set(mapping) != expected_seeds:
            raise RuntimeError(
                f"{evaluator}: expected seeds {sorted(expected_seeds)}, found {sorted(mapping)}"
            )

    results = []
    for evaluator, mapping in learned_sets.items():
        for seed in sorted(mapping):
            learned_path = mapping[seed]
            learned = read_rows(learned_path)
            results.append(
                compare(seed, evaluator, learned_path, learned, "native_c4", args.c4_csv, c4)
            )
            if no_selector is not None:
                results.append(
                    compare(
                        seed,
                        evaluator,
                        learned_path,
                        learned,
                        "fine_tune_only_no_selector",
                        args.no_selector_csv,
                        no_selector,
                    )
                )

    payload = {
        "test": "exact paired McNemar, two-sided",
        "clips_per_comparison": 1128,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

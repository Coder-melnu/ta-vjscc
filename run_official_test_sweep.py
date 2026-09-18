"""Preflight and launch the frozen one-shot official UCF101 test sweep."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CONFIRMATION = "LAUNCH_FROZEN_OFFICIAL_TEST_SWEEP"
TSN_CKPT = Path(
    "downstream/action_recognition/weights/tsn_ucf101_head_locked_split_best.pt"
)
R2_CKPT = Path(
    "downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inventory",
        type=Path,
        default=Path("official_test_sweep_checkpoint_inventory.json"),
    )
    p.add_argument(
        "--output_root", type=Path, default=Path("out_official_test_sweep_locked")
    )
    p.add_argument("--launch", action="store_true")
    p.add_argument("--confirm", default="")
    return p.parse_args()


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def command_record(label, command):
    return {"label": label, "command": [str(value) for value in command]}


def build_commands(records, output_root):
    python = sys.executable
    common = [
        "--split", "test",
        "--split_seed", "42",
        "--test_gop_seed", "44",
        "--channel_seed", "1042",
        "--batch_size", "8",
        "--num_workers", "4",
    ]
    commands = []
    baselines = sorted(
        (r for r in records if r["family"] == "corrected_baseline"),
        key=lambda r: (r["channel"], r["c"], r["snr_db"]),
    )
    for record in baselines:
        label = f"baseline_{record['channel']}_c{record['c']}_snr{record['snr_db']}"
        command = [
            python,
            "eval_locked_controls.py",
            "--model_type", "clean",
            "--videojscc_ckpt", record["path"],
            "--c", str(record["c"]),
            "--tsn_ckpt", str(TSN_CKPT),
            "--r2plus1d_ckpt", str(R2_CKPT),
            "--out", str(output_root / "baselines"),
            *common,
        ]
        commands.append(command_record(label, command))

    no_selector = [
        r for r in records if r["family"] == "fine_tune_only_no_selector"
    ]
    if len(no_selector) != 1:
        raise RuntimeError("Inventory must contain exactly one fine-tune-only record")
    record = no_selector[0]
    commands.append(command_record(
        "fine_tune_only_no_selector_c8_snr13",
        [
            python,
            "eval_locked_controls.py",
            "--model_type", "no_selector",
            "--videojscc_ckpt", record["path"],
            "--c", "8",
            "--channel", "AWGN",
            "--snr", "13",
            "--tsn_ckpt", str(TSN_CKPT),
            "--r2plus1d_ckpt", str(R2_CKPT),
            "--out", str(output_root / "fine_tune_only"),
            *common,
        ],
    ))

    topk = sorted(
        (r for r in records if r["family"] == "exact_topk"),
        key=lambda r: r["training_seed"],
    )
    if [r["training_seed"] for r in topk] != [42, 43, 44]:
        raise RuntimeError("Inventory must contain exact top-k seeds 42, 43, and 44")
    for record in topk:
        seed = record["training_seed"]
        base_args = [
            "--selector_ckpt", record["path"],
            "--tsn_ckpt", str(TSN_CKPT),
            "--split", "test",
            "--split_seed", "42",
            "--test_gop_seed", "44",
            "--channel_seed", "1042",
            "--channel", "AWGN",
            "--snr", "13",
            "--c", "8",
            "--ratio", str(1.0 / 6.0),
            "--hidden_dim", "16",
            "--tau", "0.5",
            "--random_mask_seed", "1729",
            "--keep_fraction", "0.5",
            "--batch_size", "8",
            "--num_workers", "4",
        ]
        commands.append(command_record(
            f"topk_seed{seed}_tsn_all_modes",
            [
                python,
                "downstream/action_recognition/eval_topk_selector_matched_tsn.py",
                *base_args,
                "--out", str(output_root / "topk" / f"seed{seed}" / "tsn"),
            ],
        ))
        commands.append(command_record(
            f"topk_seed{seed}_r2plus1d_all_modes",
            [
                python,
                "downstream/action_recognition/eval_topk_selector_matched_r2plus1d.py",
                *base_args,
                "--r2plus1d_ckpt", str(R2_CKPT),
                "--out", str(output_root / "topk" / f"seed{seed}" / "r2plus1d"),
            ],
        ))
    return commands


def main():
    args = parser()
    require_file(args.inventory, "checkpoint inventory")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("status") != "PASS" or inventory.get("official_test_accessed") is not False:
        raise RuntimeError("Inventory is not a passing pre-test inventory")
    if inventory.get("baseline_count") != 20:
        raise RuntimeError("Inventory does not certify exactly 20 baselines")

    required_scripts = [
        Path("eval_locked_controls.py"),
        Path("downstream/action_recognition/eval_topk_selector_matched_tsn.py"),
        Path("downstream/action_recognition/eval_topk_selector_matched_r2plus1d.py"),
    ]
    for path in required_scripts:
        require_file(path, "evaluation script")
    require_file(TSN_CKPT, "locked TSN checkpoint")
    require_file(R2_CKPT, "locked R(2+1)D checkpoint")

    for record in inventory["records"]:
        path = Path(record["path"])
        require_file(path, "frozen model checkpoint")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")

    commands = build_commands(inventory["records"], args.output_root)
    if len(commands) != 27:
        raise RuntimeError(f"Expected 27 evaluator commands, built {len(commands)}")
    plan = {
        "status": "PREFLIGHT_PASS",
        "official_test_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory),
        "inventory_sha256": sha256_file(args.inventory),
        "output_root": str(args.output_root),
        "protocol": {
            "official_split": "UCF101 split 1 testlist01.txt",
            "expected_test_clips": 3783,
            "split_seed": 42,
            "test_gop_seed": 44,
            "channel_seed": 1042,
            "random_mask_seed": 1729,
            "gop_size": 5,
            "gops_per_clip": 1,
            "image_size": 128,
            "parameter_updates": 0,
            "checkpoint_selection_on_test": False,
        },
        "evaluator_sha256": {
            str(path): sha256_file(path) for path in required_scripts
        },
        "recognizer_sha256": {
            str(TSN_CKPT): sha256_file(TSN_CKPT),
            str(R2_CKPT): sha256_file(R2_CKPT),
        },
        "command_count": len(commands),
        "commands": commands,
    }
    Path("official_test_sweep_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k != "commands"}, indent=2))
    print("Saved: official_test_sweep_plan.json")

    if not args.launch:
        print("PREFLIGHT ONLY: official test set was not accessed")
        return
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"Launch requires --confirm {CONFIRMATION}")
    if args.output_root.exists():
        raise FileExistsError(
            f"One-shot output root already exists; refusing to overwrite: {args.output_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=False)
    launch_record = {
        **plan,
        "status": "RUNNING",
        "official_test_accessed": True,
        "launch_utc": datetime.now(timezone.utc).isoformat(),
        "completed_commands": [],
    }
    run_record_path = args.output_root / "one_shot_run_record.json"
    run_record_path.write_text(json.dumps(launch_record, indent=2))
    log_path = args.output_root / "one_shot_master.log"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."

    with log_path.open("w") as log:
        for index, item in enumerate(commands, start=1):
            header = f"[{index}/{len(commands)}] {item['label']}"
            print(header)
            log.write(header + "\n")
            log.flush()
            process = subprocess.Popen(
                item["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
            if return_code != 0:
                launch_record["status"] = "FAILED"
                launch_record["failed_command"] = item
                launch_record["return_code"] = return_code
                run_record_path.write_text(json.dumps(launch_record, indent=2))
                raise RuntimeError(f"Official sweep stopped at {item['label']}")
            launch_record["completed_commands"].append(item["label"])
            run_record_path.write_text(json.dumps(launch_record, indent=2))

    launch_record["status"] = "COMPLETED"
    launch_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_record_path.write_text(json.dumps(launch_record, indent=2))
    print(f"Official sweep completed: {args.output_root}")


if __name__ == "__main__":
    main()

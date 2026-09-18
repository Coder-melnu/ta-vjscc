"""Create convergence plots for one audited clean VideoJSCC run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metric rows found in {path}")
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Completed VideoJSCC run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    values = read_metrics(run_dir / "metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    best_epoch = int(summary["best_epoch"])
    best_index = values["epoch"].index(float(best_epoch))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(run_dir.name.replace("_", " "), fontsize=12)

    axes[0, 0].plot(values["epoch"], values["train_loss"], label="Train loss")
    axes[0, 0].plot(values["epoch"], values["val_loss"], label="Validation loss")
    axes[0, 0].scatter(best_epoch, values["val_loss"][best_index], marker="*", s=110,
                       label=f"Selected epoch {best_epoch}")
    axes[0, 0].set(title="Reconstruction loss", xlabel="Epoch", ylabel="MSE")
    axes[0, 0].legend()

    axes[0, 1].plot(values["epoch"], values["val_psnr_db"])
    axes[0, 1].scatter(best_epoch, values["val_psnr_db"][best_index], marker="*", s=110)
    axes[0, 1].set(title="Validation PSNR", xlabel="Epoch", ylabel="dB")

    axes[1, 0].plot(values["epoch"], values["val_ssim"])
    axes[1, 0].scatter(best_epoch, values["val_ssim"][best_index], marker="*", s=110)
    axes[1, 0].set(title="Validation SSIM", xlabel="Epoch", ylabel="SSIM")

    axes[1, 1].plot(values["epoch"], values["val_ms_ssim_3scale"])
    axes[1, 1].scatter(best_epoch, values["val_ms_ssim_3scale"][best_index], marker="*", s=110)
    axes[1, 1].set(title="Validation MS-SSIM (3-scale)", xlabel="Epoch", ylabel="MS-SSIM")

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    output = run_dir / "convergence.png"
    figure.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()

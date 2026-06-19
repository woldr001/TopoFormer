#!/usr/bin/env python3
"""Plot per-channel heatmaps (filtration step x side-chain combo) for a
side-chain centroid topology feature file.

Usage
-----
    python protein_function/scripts/plot_sidechain_heatmaps.py \\
        --npy_path /mnt/research/nodes/Woldring/topo_features_sidechain/A0A031WDA8.npy \\
        --output   A0A031WDA8_heatmaps.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from protein_function.topo_extraction.sidechain_topo_embedding import (
    CLASS_LABELS,
    SIDECHAIN_COMBINATIONS,
)

STAT_NAMES = ["count_zero", "max", "sum", "nonzero_mean", "nonzero_std", "nonzero_min"]
CHANNEL_NAMES = [f"mean({s})" for s in STAT_NAMES] + [f"std({s})" for s in STAT_NAMES]


def combo_label(combo):
    return "+".join(CLASS_LABELS[c] for c in combo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy_path", required=True, help="Path to a [12, 200, 15] .npy file.")
    parser.add_argument("--output", default=None, help="Save figure to this path instead of showing it.")
    parser.add_argument("--dis_start", type=float, default=0.0)
    parser.add_argument("--dis_step", type=float, default=0.2)
    args = parser.parse_args()

    arr = np.load(args.npy_path)
    if arr.ndim != 3 or arr.shape[0] != len(CHANNEL_NAMES):
        raise ValueError(
            f"Expected shape (12, n_filtrations, 15), got {arr.shape}. "
            "Check that this file was produced by the sidechain_centroid mode."
        )
    n_channels, n_filt, n_combos = arr.shape
    distances = args.dis_start + np.arange(n_filt) * args.dis_step
    combo_labels = [combo_label(c) for c in SIDECHAIN_COMBINATIONS]

    n_cols = 4
    n_rows = -(-n_channels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i in range(n_channels):
        ax = axes[i // n_cols][i % n_cols]
        im = ax.imshow(
            arr[i],
            aspect="auto",
            origin="lower",
            extent=[0, n_combos, distances[0], distances[-1]],
            cmap="viridis",
        )
        ax.set_title(CHANNEL_NAMES[i], fontsize=9)
        ax.set_xticks(np.arange(n_combos) + 0.5)
        ax.set_xticklabels(combo_labels, rotation=90, fontsize=6)
        ax.set_ylabel("distance (Å)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_channels, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    fig.suptitle(args.npy_path, fontsize=10)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

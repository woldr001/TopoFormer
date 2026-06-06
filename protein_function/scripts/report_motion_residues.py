#!/usr/bin/env python3
"""Report motion vs. static residue counts for every protein in the NMA-PCA directory.

Applies the same scoring and thresholding logic as the topo feature extractor
(load_motion_scores + identify_motion_residues) so the counts match exactly what
is used during topology feature computation.

Output columns
--------------
protein_id      UniProt accession
n_residues      Total residues in the GNM/PCA profile
n_motion        Residues classified as high-motion (top_motion_pct)
n_static        Residues classified as static
pct_motion      n_motion / n_residues * 100
sources         Which profile files were found (gnm, pca, or both)

Usage
-----
    python report_motion_residues.py \\
        --nma_pca_dir  /path/to/nma_pca \\
        --id_file      /path/to/all_ids.txt \\
        --output       motion_residue_report.tsv \\
        --top_motion_pct 0.20 \\
        --min_motion_residues 10 \\
        --n_slow_modes 3
"""

import argparse
import os
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Score loading (mirrors ensemble_motion_topo_embedding.load_motion_scores)
# ---------------------------------------------------------------------------

def load_motion_scores(nma_pca_dir, fluctuation_sources=("gnm", "pca"), n_slow_modes=3):
    score_arrays = []
    resids = None
    sources_found = []

    if "gnm" in fluctuation_sources:
        gnm_path = os.path.join(nma_pca_dir, "anm_gnm_results.npz")
        if os.path.exists(gnm_path):
            data = np.load(gnm_path, allow_pickle=True)
            if "gnm_flucts_mean" in data:
                flucts = data["gnm_flucts_mean"][:n_slow_modes]
                gnm_score = flucts.sum(axis=0)
                r = gnm_score.max() - gnm_score.min()
                if r > 0:
                    gnm_score = (gnm_score - gnm_score.min()) / r
                score_arrays.append(gnm_score)
                sources_found.append("gnm")
                if "resids" in data and resids is None:
                    resids = data["resids"].astype(int)

    if "pca" in fluctuation_sources:
        pca_path = os.path.join(nma_pca_dir, "pca_results.npz")
        if os.path.exists(pca_path):
            data = np.load(pca_path, allow_pickle=True)
            n = len(data["resids"]) if "resids" in data else 0
            pc_score = np.zeros(n, dtype=np.float64)
            for pc_key in ("pc1_profile", "pc2_profile"):
                if pc_key in data:
                    pc = data[pc_key].astype(np.float64)
                    r = pc.max() - pc.min()
                    if r > 0:
                        pc = (pc - pc.min()) / r
                    pc_score += pc
            if pc_score.size > 0:
                r = pc_score.max() - pc_score.min()
                if r > 0:
                    pc_score = (pc_score - pc_score.min()) / r
                score_arrays.append(pc_score)
                sources_found.append("pca")
            if "resids" in data and resids is None:
                resids = data["resids"].astype(int)

    if not score_arrays:
        return None, None, []

    combined = np.sum(score_arrays, axis=0).astype(np.float64)
    return combined, resids, sources_found


def identify_motion_residues(scores, resids, top_pct=0.20, min_residues=10):
    n_select = max(min_residues, int(np.ceil(len(scores) * top_pct)))
    n_select = min(n_select, len(scores))
    top_idx = np.argsort(scores)[::-1][:n_select]
    return resids[top_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Report motion/static residue counts per protein from NMA-PCA outputs."
    )
    parser.add_argument(
        "--nma_pca_dir", required=True,
        help="Root directory containing one sub-directory per protein ID.",
    )
    parser.add_argument(
        "--id_file",
        help="Text file listing one protein ID per line. "
             "If omitted, every sub-directory of nma_pca_dir is scanned.",
    )
    parser.add_argument(
        "--output", default="motion_residue_report.tsv",
        help="Output TSV file path. Default: motion_residue_report.tsv",
    )
    parser.add_argument(
        "--top_motion_pct", type=float, default=0.20,
        help="Fraction of residues treated as high-motion. Default: 0.20",
    )
    parser.add_argument(
        "--min_motion_residues", type=int, default=10,
        help="Minimum number of motion residues regardless of top_motion_pct. Default: 10",
    )
    parser.add_argument(
        "--n_slow_modes", type=int, default=3,
        help="Number of GNM slow modes to sum. Default: 3",
    )
    parser.add_argument(
        "--fluctuation_sources", nargs="+", default=["gnm", "pca"],
        choices=["gnm", "pca"],
        help="Which profile files to combine. Default: gnm pca",
    )
    args = parser.parse_args()

    # Build protein list
    if args.id_file:
        with open(args.id_file) as fh:
            protein_ids = [l.strip() for l in fh if l.strip()]
    else:
        protein_ids = sorted(
            d for d in os.listdir(args.nma_pca_dir)
            if os.path.isdir(os.path.join(args.nma_pca_dir, d))
        )

    print(f"Reporting motion residues for {len(protein_ids)} proteins ...")

    rows = []
    n_missing = 0
    n_ok = 0

    for pid in protein_ids:
        pdir = os.path.join(args.nma_pca_dir, pid)
        if not os.path.isdir(pdir):
            rows.append({
                "protein_id": pid, "n_residues": "NA", "n_motion": "NA",
                "n_static": "NA", "pct_motion": "NA", "sources": "missing_dir",
            })
            n_missing += 1
            continue

        scores, resids, sources = load_motion_scores(
            pdir,
            fluctuation_sources=args.fluctuation_sources,
            n_slow_modes=args.n_slow_modes,
        )

        if scores is None:
            rows.append({
                "protein_id": pid, "n_residues": "NA", "n_motion": "NA",
                "n_static": "NA", "pct_motion": "NA", "sources": "no_profiles",
            })
            n_missing += 1
            continue

        motion_resids = identify_motion_residues(
            scores, resids,
            top_pct=args.top_motion_pct,
            min_residues=args.min_motion_residues,
        )

        n_res    = len(scores)
        n_motion = len(motion_resids)
        n_static = n_res - n_motion
        pct      = n_motion / n_res * 100 if n_res > 0 else 0.0

        rows.append({
            "protein_id": pid,
            "n_residues": n_res,
            "n_motion":   n_motion,
            "n_static":   n_static,
            "pct_motion": f"{pct:.1f}",
            "sources":    "+".join(sources),
        })
        n_ok += 1

    # Write TSV
    header = ["protein_id", "n_residues", "n_motion", "n_static", "pct_motion", "sources"]
    with open(args.output, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in header) + "\n")

    print(f"Written: {args.output}")
    print(f"  OK:      {n_ok}")
    print(f"  Missing: {n_missing}")

    # Summary statistics
    valid = [r for r in rows if r["n_residues"] != "NA"]
    if valid:
        n_res_vals    = [int(r["n_residues"]) for r in valid]
        n_motion_vals = [int(r["n_motion"])   for r in valid]
        n_static_vals = [int(r["n_static"])   for r in valid]
        print(f"\nSummary across {len(valid)} proteins:")
        print(f"  Residues  — min: {min(n_res_vals):4d}  mean: {sum(n_res_vals)/len(n_res_vals):6.1f}  max: {max(n_res_vals):4d}")
        print(f"  Motion    — min: {min(n_motion_vals):4d}  mean: {sum(n_motion_vals)/len(n_motion_vals):6.1f}  max: {max(n_motion_vals):4d}")
        print(f"  Static    — min: {min(n_static_vals):4d}  mean: {sum(n_static_vals)/len(n_static_vals):6.1f}  max: {max(n_static_vals):4d}")
        pct_vals = [float(r["pct_motion"]) for r in valid]
        print(f"  Pct motion— min: {min(pct_vals):5.1f}%  mean: {sum(pct_vals)/len(pct_vals):5.1f}%  max: {max(pct_vals):5.1f}%")


if __name__ == "__main__":
    main()

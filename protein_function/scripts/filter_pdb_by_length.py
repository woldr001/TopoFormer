#!/usr/bin/env python3
"""Filter PDB files by protein length and copy qualifying structures to a subdirectory.

Usage
-----
    python filter_pdb_by_length.py \
        --source_dir /mnt/research/nodes/trainingset/afdatabase \
        --dest_dir   /mnt/research/nodes/trainingset/afdatabase/sub_500aa \
        --max_aa     500 \
        --n_workers  16

The script counts residues from SEQRES records when present (the total
chain length, including any unmodelled residues).  For files without
SEQRES it falls back to counting unique (chain, resSeq, iCode) tuples
from ATOM records (modelled residues only).  AlphaFold DB PDB files
always include SEQRES, so the primary path is used for those.

Outputs
-------
- Copies qualifying .pdb files to <dest_dir>/
- Writes a TSV report to <dest_dir>/length_report.tsv with columns:
      filename  n_residues  copied  error
- Prints a summary to stdout when finished.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
from multiprocessing import Pool
from functools import partial


# ---------------------------------------------------------------------------
# Residue counting
# ---------------------------------------------------------------------------

def count_residues_seqres(lines):
    """Sum residue counts from SEQRES records (one total per chain).

    The SEQRES format repeats the chain total on every line for that chain.
    We keep one value per chain (the first seen) and sum across chains.

    Returns total residue count, or 0 if no SEQRES records found.
    """
    chain_counts = {}
    for line in lines:
        if not line.startswith("SEQRES"):
            continue
        chain = line[11] if len(line) > 11 else " "
        try:
            count = int(line[13:17])
        except (ValueError, IndexError):
            continue
        if chain not in chain_counts:
            chain_counts[chain] = count
    return sum(chain_counts.values())


def count_residues_atom(lines):
    """Count unique (chain, resSeq, iCode) tuples from ATOM records.

    Used as fallback when SEQRES is absent.
    """
    residues = set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        try:
            chain  = line[21]
            resseq = line[22:26].strip()
            icode  = line[26]
            residues.add((chain, resseq, icode))
        except IndexError:
            continue
    return len(residues)


def count_residues(pdb_path):
    """Return the amino acid count for a PDB file.

    Prefers SEQRES (total chain length); falls back to ATOM unique residues.
    """
    with open(pdb_path, errors="replace") as fh:
        lines = fh.readlines()

    n = count_residues_seqres(lines)
    if n > 0:
        return n, "seqres"

    n = count_residues_atom(lines)
    return n, "atom"


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def process_one(pdb_path, dest_dir, max_aa):
    """Count residues; copy to dest_dir if <= max_aa.

    Returns a dict with keys: filename, n_residues, method, copied, error.
    """
    fname = os.path.basename(pdb_path)
    result = dict(filename=fname, n_residues=-1, method="", copied=False, error="")

    try:
        n, method = count_residues(pdb_path)
        result["n_residues"] = n
        result["method"]     = method
    except Exception as exc:
        result["error"] = str(exc)
        return result

    if result["n_residues"] <= max_aa:
        dest = os.path.join(dest_dir, fname)
        try:
            shutil.copy2(pdb_path, dest)
            result["copied"] = True
        except Exception as exc:
            result["error"] = f"copy failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Copy PDB files with ≤ max_aa residues to a subdirectory."
    )
    parser.add_argument(
        "--source_dir",
        default="/mnt/research/nodes/trainingset/afdatabase",
        help="Directory containing input *.pdb files.",
    )
    parser.add_argument(
        "--dest_dir",
        default="/mnt/research/nodes/trainingset/afdatabase/sub_500aa",
        help="Output directory for qualifying PDB files.",
    )
    parser.add_argument(
        "--max_aa",
        type=int,
        default=500,
        help="Maximum protein length in amino acids (inclusive). Default: 500.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes. Default: 8.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdb",
        help="Glob pattern for PDB files. Default: '*.pdb'.",
    )
    args = parser.parse_args()

    # Validate source
    if not os.path.isdir(args.source_dir):
        sys.exit(f"ERROR: source_dir does not exist: {args.source_dir}")

    os.makedirs(args.dest_dir, exist_ok=True)

    pdb_files = sorted(
        glob.glob(os.path.join(args.source_dir, args.pattern))
    )

    # Exclude files already inside dest_dir (handles nested case)
    dest_abs = os.path.abspath(args.dest_dir)
    pdb_files = [
        p for p in pdb_files
        if not os.path.abspath(p).startswith(dest_abs + os.sep)
    ]

    total = len(pdb_files)
    if total == 0:
        sys.exit(f"No files matching '{args.pattern}' found in {args.source_dir}")

    print(f"Source : {args.source_dir}")
    print(f"Dest   : {args.dest_dir}")
    print(f"Filter : length ≤ {args.max_aa} aa")
    print(f"Workers: {args.n_workers}")
    print(f"Files  : {total:,}")
    print()

    report_path = os.path.join(args.dest_dir, "length_report.tsv")
    report_rows = []

    worker = partial(process_one, dest_dir=args.dest_dir, max_aa=args.max_aa)

    n_copied  = 0
    n_skipped = 0
    n_errors  = 0
    log_every = max(1, total // 20)   # ~5% intervals

    with Pool(processes=args.n_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(worker, pdb_files), start=1
        ):
            report_rows.append(result)

            if result["error"]:
                n_errors += 1
            elif result["copied"]:
                n_copied += 1
            else:
                n_skipped += 1

            if i % log_every == 0 or i == total:
                pct = 100 * i / total
                print(
                    f"  {i:>{len(str(total))}}/{total}  ({pct:5.1f}%)  "
                    f"copied: {n_copied:,}  skipped(>{args.max_aa}aa): {n_skipped:,}  "
                    f"errors: {n_errors}"
                )

    # Write TSV report
    report_rows.sort(key=lambda r: r["filename"])
    with open(report_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["filename", "n_residues", "method", "copied", "error"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 60)
    print(f"  Copied  (≤{args.max_aa} aa) : {n_copied:,}")
    print(f"  Skipped (>{args.max_aa} aa) : {n_skipped:,}")
    print(f"  Errors               : {n_errors}")
    print(f"  Report saved to      : {report_path}")
    print("=" * 60)

    if n_errors:
        print("\nFirst 10 errors:")
        err_rows = [r for r in report_rows if r["error"]]
        for r in err_rows[:10]:
            print(f"  {r['filename']}: {r['error']}")


if __name__ == "__main__":
    main()

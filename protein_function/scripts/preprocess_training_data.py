#!/usr/bin/env python3
"""Prepare the training dataset for TopoFormer-MF.

Handles the one-time data-wrangling needed before running the topology
and sequence precompute scripts.  Run this once; everything it produces
is a prerequisite for every subsequent step.

What this script does
---------------------
1. Scans <sampling_dir> to discover proteins with complete aSAM ensembles
   (10 raw.seed_*.pdb files under clustering.nodes_pipeline_v1/).
2. Parses the MF-annotation.txt (comma-separated, Python-list GO terms)
   → long-format TSV expected by build_mf_label_matrix().
3. Parses sequences.fasta → filtered FASTA containing only proteins
   present in all three sources (ensembles + annotations + sequences).
4. Reports ID-format mismatches (UniProt accession vs PDB/mnemonic).
5. Creates a manifest CSV for QA.
6. Creates train / val / test split ID files (random 70/15/15 by default;
   swap in CD-HIT cluster files for a proper sequence-identity split).

Usage
-----
    python preprocess_training_data.py \\
        --sampling_dir   /mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling \\
        --annotation_file /mnt/research/nodes/GOAnnotation/MF-annotation.txt \\
        --fasta_file     /mnt/research/nodes/testsequences/seq.fasta \\
        --output_dir     /mnt/research/woldring_lab/TopoFormer-MF/datasets \\
        --train_frac 0.70 --val_frac 0.15 --seed 42

Output directory layout
-----------------------
    datasets/
    ├── mf_annotations.tsv       # long-format: protein_id  go_term  namespace
    ├── sequences.fasta          # FASTA filtered to proteins with all data
    ├── train_ids.txt
    ├── val_ids.txt
    ├── test_ids.txt
    ├── manifest.csv             # one row per protein, data-completeness flags
    └── id_mismatch_report.txt   # proteins present in annotations/FASTA but
                                 # not found in sampling_dir (ID format issues)
"""

import argparse
import ast
import csv
import glob
import os
import random
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASAM_SUBDIR  = "clustering.nodes_pipeline_v1"
ASAM_PATTERN = "raw.seed_*.pdb"
N_CONFORMERS = 10

# UniProt accession regex (6-char and 10-char forms).
# Ref: https://www.uniprot.org/help/accession_numbers
#
# 6-char rules:
#   O/P/Q prefix → [OPQ][0-9][A-Z0-9]{3}[0-9]   (pos 3-5 may be digit or letter)
#   other prefix → [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]  (pos 3 must be a letter)
# 10-char rule (only [A-NR-Z] prefix):
#   [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){2}
_UNIPROT_AC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"                              # 6-char, O/P/Q
    r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]"                         # 6-char, other
    r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9])$"  # 10-char
)


def is_uniprot_accession(protein_id: str) -> bool:
    """Return True if protein_id looks like a UniProt accession.

    Rejects PDB+chain IDs (e.g. '6c3u-A', '3gtu-B') and UniProt
    mnemonic names (e.g. 'ADH2_YEAST') — both are excluded from training.
    """
    return bool(_UNIPROT_AC_RE.match(protein_id))


def discover_ensemble_proteins(sampling_dir):
    """Return dict {protein_id: [pdb_path, ...]} for proteins with ≥1 PDB.

    Expected layout:
        <sampling_dir>/<protein_id>/clustering.nodes_pipeline_v1/raw.seed_*.pdb
    """
    result = {}
    for entry in sorted(os.scandir(sampling_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        pid = entry.name
        pattern = os.path.join(entry.path, ASAM_SUBDIR, ASAM_PATTERN)
        pdbs = sorted(glob.glob(pattern))
        if pdbs:
            result[pid] = pdbs
    return result


def parse_annotation_file(annotation_file):
    """Parse MF-annotation.txt → dict {protein_id: [go_term, ...]}.

    Only retains entries whose ID is a valid UniProt accession.
    PDB+chain IDs (e.g. '6c3u-A') and UniProt mnemonic names
    (e.g. 'ADH2_YEAST') are silently skipped and counted.

    Input format (either works):
        P32234,['GO:0003924', 'GO:0005525']
        6c3u-A,['GO:0051213']        ← skipped (PDB+chain)
    """
    annot          = {}
    n_skipped_pdb  = 0
    n_skipped_other = 0
    go_re = re.compile(r"GO:\d+")

    with open(annotation_file) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            comma = line.find(",")
            if comma == -1:
                print(f"  [warn] line {lineno}: no comma, skipping: {line!r}")
                continue

            pid      = line[:comma].strip()
            go_field = line[comma + 1:].strip()

            if not is_uniprot_accession(pid):
                if re.match(r"^\d\w{3}-\w+$", pid):   # looks like PDB+chain
                    n_skipped_pdb += 1
                else:
                    n_skipped_other += 1
                continue

            terms = go_re.findall(go_field)
            if not terms:
                print(f"  [warn] line {lineno}: no GO terms for {pid!r}: {go_field!r}")
                continue

            annot[pid] = terms

    print(f"    Skipped (PDB+chain IDs):  {n_skipped_pdb:,}")
    print(f"    Skipped (other non-AC):   {n_skipped_other:,}")
    return annot


def parse_fasta(fasta_file):
    """Return dict {protein_id: sequence}, keeping only UniProt accessions.

    Accepts both standard multi-line FASTA and single-line variants.
    The protein_id is the first whitespace-delimited token after '>'.
    Entries whose ID is not a valid UniProt accession (e.g. PDB+chain
    IDs like '1oat-A') are skipped and counted.
    """
    seqs           = {}
    current_id     = None
    current_seq    = []
    n_skipped_pdb  = 0
    n_skipped_other = 0

    def _flush():
        nonlocal current_id, current_seq
        if current_id is not None and is_uniprot_accession(current_id):
            seqs[current_id] = "".join(current_seq)
        current_id  = None
        current_seq = []

    with open(fasta_file) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                raw_id = line[1:].split()[0]
                if not is_uniprot_accession(raw_id):
                    if re.match(r"^\d\w{3}-\w+$", raw_id):
                        n_skipped_pdb += 1
                    else:
                        n_skipped_other += 1
                    current_id = None  # mark as skip-this-entry
                else:
                    current_id = raw_id
                current_seq = []
            else:
                current_seq.append(line)

    _flush()

    print(f"    Skipped FASTA entries (PDB+chain IDs): {n_skipped_pdb:,}")
    print(f"    Skipped FASTA entries (other non-AC):  {n_skipped_other:,}")
    return seqs


def write_annotation_tsv(annot_dict, protein_ids, out_path):
    """Write long-format TSV: protein_id \\t go_term \\t namespace."""
    with open(out_path, "w") as fh:
        fh.write("protein_id\tgo_term\tnamespace\n")
        for pid in sorted(protein_ids):
            for term in annot_dict.get(pid, []):
                fh.write(f"{pid}\t{term}\tmolecular_function\n")
    print(f"  Wrote annotation TSV → {out_path}")


def write_fasta(seq_dict, protein_ids, out_path):
    """Write FASTA for the given protein IDs."""
    with open(out_path, "w") as fh:
        for pid in sorted(protein_ids):
            if pid in seq_dict:
                fh.write(f">{pid}\n{seq_dict[pid]}\n")
    print(f"  Wrote FASTA → {out_path}")


def write_id_list(protein_ids, out_path):
    with open(out_path, "w") as fh:
        for pid in sorted(protein_ids):
            fh.write(pid + "\n")


def random_split(protein_ids, train_frac, val_frac, seed):
    ids = sorted(protein_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train   = ids[:n_train]
    val     = ids[n_train:n_train + n_val]
    test    = ids[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TopoFormer-MF training dataset from aSAM ensembles."
    )
    parser.add_argument(
        "--sampling_dir",
        required=True,
        help="Root aSAM sampling directory containing one sub-dir per protein.",
    )
    parser.add_argument(
        "--annotation_file",
        required=True,
        help="MF-annotation.txt (comma-separated, Python-list GO terms).",
    )
    parser.add_argument(
        "--fasta_file",
        required=True,
        help="FASTA file with protein sequences.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where all dataset files will be written.",
    )
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--min_conformers",
        type=int,
        default=10,
        help="Minimum number of PDB conformers required. Default: 10.",
    )
    parser.add_argument(
        "--min_go_terms",
        type=int,
        default=1,
        help="Minimum number of GO terms required. Default: 1.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover ensemble proteins
    # ------------------------------------------------------------------
    print(f"\n[1] Scanning ensemble directory: {args.sampling_dir}")
    ensemble_map = discover_ensemble_proteins(args.sampling_dir)
    print(f"    Found {len(ensemble_map):,} protein directories with aSAM PDB files.")

    # Flag proteins with fewer than expected conformers
    incomplete = {
        pid: pdbs for pid, pdbs in ensemble_map.items()
        if len(pdbs) < args.min_conformers
    }
    if incomplete:
        print(f"    WARNING: {len(incomplete)} proteins have <{args.min_conformers} conformers:")
        for pid, pdbs in list(incomplete.items())[:10]:
            print(f"      {pid}: {len(pdbs)} PDB files")
        if len(incomplete) > 10:
            print(f"      ... and {len(incomplete)-10} more")

    # Only keep proteins meeting the conformer requirement
    ensemble_ids = set(
        pid for pid, pdbs in ensemble_map.items()
        if len(pdbs) >= args.min_conformers
    )
    print(f"    Proteins with ≥{args.min_conformers} conformers: {len(ensemble_ids):,}")

    # ------------------------------------------------------------------
    # 2. Parse annotation file
    # ------------------------------------------------------------------
    print(f"\n[2] Parsing annotation file: {args.annotation_file}")
    annot_dict = parse_annotation_file(args.annotation_file)
    print(f"    Loaded annotations for {len(annot_dict):,} proteins.")

    # Filter by minimum GO terms
    annot_dict = {
        pid: terms for pid, terms in annot_dict.items()
        if len(terms) >= args.min_go_terms
    }
    annotation_ids = set(annot_dict.keys())
    print(f"    Proteins with ≥{args.min_go_terms} GO terms: {len(annotation_ids):,}")

    # ------------------------------------------------------------------
    # 3. Parse FASTA
    # ------------------------------------------------------------------
    print(f"\n[3] Parsing FASTA: {args.fasta_file}")
    seq_dict   = parse_fasta(args.fasta_file)
    fasta_ids  = set(seq_dict.keys())
    print(f"    Loaded sequences for {len(fasta_ids):,} proteins.")

    # ------------------------------------------------------------------
    # 4. Find intersection and report mismatches
    # ------------------------------------------------------------------
    print("\n[4] Cross-referencing all three sources ...")

    complete_ids = ensemble_ids & annotation_ids & fasta_ids

    print(f"    Proteins with ensembles:              {len(ensemble_ids):,}")
    print(f"    Proteins with GO annotations:         {len(annotation_ids):,}")
    print(f"    Proteins with sequences:              {len(fasta_ids):,}")
    print(f"    >>> Proteins with ALL three:          {len(complete_ids):,}")

    # Report missing
    in_ensemble_not_annot  = ensemble_ids - annotation_ids
    in_ensemble_not_fasta  = ensemble_ids - fasta_ids
    in_annot_not_ensemble  = annotation_ids - ensemble_ids
    in_fasta_not_ensemble  = fasta_ids - ensemble_ids

    mismatch_path = os.path.join(args.output_dir, "id_mismatch_report.txt")
    with open(mismatch_path, "w") as fh:
        fh.write("# ID mismatch report\n")
        fh.write("# Proteins in ensemble dir but NOT in annotation file:\n")
        for pid in sorted(in_ensemble_not_annot):
            fh.write(f"  ENSEMBLE_ONLY  {pid}\n")
        fh.write("\n# Proteins in ensemble dir but NOT in FASTA:\n")
        for pid in sorted(in_ensemble_not_fasta):
            fh.write(f"  NO_SEQUENCE    {pid}\n")
        fh.write("\n# Proteins in annotation file but NOT in ensemble dir:\n")
        fh.write("# (likely ID format mismatch — annotation uses PDB/mnemonic, dir uses UniProt accession)\n")
        for pid in sorted(in_annot_not_ensemble):
            fh.write(f"  ANNOT_ONLY     {pid}\n")
        fh.write("\n# Proteins in FASTA but NOT in ensemble dir:\n")
        for pid in sorted(in_fasta_not_ensemble):
            fh.write(f"  FASTA_ONLY     {pid}\n")

    print(f"\n    ID mismatch report → {mismatch_path}")
    print(f"    Ensemble proteins missing from annotations: {len(in_ensemble_not_annot):,}")
    print(f"    Ensemble proteins missing from FASTA:       {len(in_ensemble_not_fasta):,}")
    print(f"    Annotation proteins not in ensemble dir:    {len(in_annot_not_ensemble):,}")

    if len(in_annot_not_ensemble) > 0:
        print()
        print("    *** ID FORMAT NOTE ***")
        print("    Some annotation entries use PDB+chain IDs (e.g. 6c3u-A) or UniProt")
        print("    mnemonic names (e.g. ADH2_YEAST), but the ensemble directories use")
        print("    UniProt accession codes (e.g. A0A010).  These will NOT automatically")
        print("    match.  To resolve:")
        print("    1. Download the UniProt ID mapping table from:")
        print("       https://www.uniprot.org/id-mapping/")
        print("       (Map: 'UniProtKB AC/ID' ↔ 'Gene Name' and 'PDB')")
        print("    2. Add a --id_mapping_file argument to this script, OR")
        print("    3. Rename ensemble directories to match annotation IDs, OR")
        print("    4. Manually edit mf_annotations.txt so all IDs use UniProt accessions.")
        print("    See id_mismatch_report.txt for the full list.")

    if len(complete_ids) == 0:
        print("\nERROR: No proteins found in all three sources.")
        print("This is almost always an ID format mismatch.  See id_mismatch_report.txt.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Write annotation TSV (long format)
    # ------------------------------------------------------------------
    print(f"\n[5] Writing long-format annotation TSV ...")
    annot_tsv = os.path.join(args.output_dir, "mf_annotations.tsv")
    write_annotation_tsv(annot_dict, complete_ids, annot_tsv)

    # ------------------------------------------------------------------
    # 6. Write filtered FASTA
    # ------------------------------------------------------------------
    print(f"\n[6] Writing filtered FASTA ...")
    fasta_out = os.path.join(args.output_dir, "sequences.fasta")
    write_fasta(seq_dict, complete_ids, fasta_out)

    # ------------------------------------------------------------------
    # 7. Train / val / test split
    # ------------------------------------------------------------------
    print(f"\n[7] Creating random {args.train_frac:.0%}/{args.val_frac:.0%}/"
          f"{1-args.train_frac-args.val_frac:.0%} split (seed={args.seed}) ...")
    print("    NOTE: for publication, replace with a CD-HIT sequence-cluster split.")

    train_ids, val_ids, test_ids = random_split(
        complete_ids, args.train_frac, args.val_frac, args.seed
    )
    write_id_list(train_ids, os.path.join(args.output_dir, "train_ids.txt"))
    write_id_list(val_ids,   os.path.join(args.output_dir, "val_ids.txt"))
    write_id_list(test_ids,  os.path.join(args.output_dir, "test_ids.txt"))
    print(f"    Train: {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}")

    # ------------------------------------------------------------------
    # 8. Write manifest
    # ------------------------------------------------------------------
    print(f"\n[8] Writing manifest ...")
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    split_map = (
        {pid: "train" for pid in train_ids}
        | {pid: "val"   for pid in val_ids}
        | {pid: "test"  for pid in test_ids}
    )

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "protein_id", "n_conformers", "has_annotation", "has_sequence",
            "n_go_terms", "seq_length", "split"
        ])
        for pid in sorted(complete_ids):
            writer.writerow([
                pid,
                len(ensemble_map.get(pid, [])),
                pid in annot_dict,
                pid in seq_dict,
                len(annot_dict.get(pid, [])),
                len(seq_dict.get(pid, "")),
                split_map.get(pid, ""),
            ])
    print(f"    Manifest → {manifest_path}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  Complete proteins (all three sources): {len(complete_ids):,}")
    print(f"  Train / Val / Test:  {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  Output directory: {args.output_dir}")
    print()
    print("NEXT STEPS")
    print("-" * 62)
    print("  1. Check id_mismatch_report.txt and resolve ID format issues.")
    print("  2. Run the Ensemble_NMA-PCA pipeline (see README Step 2).")
    print("  3. Precompute topology features:")
    print(f"       python protein_function/scripts/precompute_topo_features.py \\")
    print(f"           --mode ensemble_motion \\")
    print(f"           --pdb_dir {args.sampling_dir} \\")
    print(f"           --nma_pca_dir <nma_pca_output_dir> \\")
    print(f"           --output_dir <topo_features_dir> \\")
    print(f"           --n_workers 16")
    print("  4. Precompute sequence embeddings:")
    print(f"       python protein_function/scripts/precompute_seq_features.py \\")
    print(f"           --fasta_file {fasta_out} \\")
    print(f"           --esm_output_dir <esm_dir> \\")
    print(f"           --prottrans_output_dir <prottrans_dir> \\")
    print(f"           --device cuda")
    print("  5. Train:")
    print(f"       python protein_function/training/train_mf_prediction.py \\")
    print(f"           --topo_dir <topo_features_dir> \\")
    print(f"           --esm_dir <esm_dir> \\")
    print(f"           --prottrans_dir <prottrans_dir> \\")
    print(f"           --label_file {annot_tsv} \\")
    print(f"           --train_ids_file {os.path.join(args.output_dir, 'train_ids.txt')} \\")
    print(f"           --val_ids_file {os.path.join(args.output_dir, 'val_ids.txt')} \\")
    print(f"           --topo_feature_mode ensemble_motion \\")
    print(f"           --output_dir <model_output_dir>")
    print("=" * 62)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report per-protein topology feature computation time.

For each completed .npy file:
  - Finds which SLURM task processed it (by scanning the task .out log files)
  - Gets the task start time from the .out header line (all 5 proteins in a task
    start simultaneously via multiprocessing.Pool)
  - Gets the protein's finish time from the .npy file mtime (written when the
    protein's feature array is saved)
  - Reports duration = finish_time − task_start_time

Usage
-----
    python report_protein_timing.py \\
        --topo_dir  /mnt/research/woldring_lab/TopoFormer-MF/topo_features \\
        --log_dir   /mnt/research/woldring_lab/TopoFormer-MF/logs \\
        --job_id    9361260

    # Write to file, sort by duration (slowest first)
    python report_protein_timing.py \\
        --topo_dir  /mnt/research/woldring_lab/TopoFormer-MF/topo_features \\
        --log_dir   /mnt/research/woldring_lab/TopoFormer-MF/logs \\
        --job_id    9361260 \\
        --output    protein_timing.tsv \\
        --sort      duration
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_start_time(time_str):
    """Parse 'Sat Jun  6 07:58:06 PM EDT 2026' → datetime (local, tz stripped)."""
    s = re.sub(r'\s+', ' ', time_str).strip()
    # Strip 3-4 letter timezone abbreviation immediately before the 4-digit year
    s = re.sub(r' [A-Z]{2,4} (\d{4})$', r' \1', s)
    return datetime.strptime(s, '%a %b %d %I:%M:%S %p %Y')


def fmt_duration(seconds):
    if seconds is None or seconds < 0:
        return 'n/a'
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def build_protein_map(log_dir, job_id):
    """Return {protein_id: {'task_id': int, 'start_dt': datetime|None}}."""
    pattern   = os.path.join(log_dir, f'topo_{job_id}_*.out')
    out_files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'_(\d+)\.out$', f).group(1))
    )
    if not out_files:
        print(f"ERROR: no files match {pattern}", file=sys.stderr)
        sys.exit(1)

    protein_map = {}

    for out_path in out_files:
        task_id = int(re.search(r'_(\d+)\.out$', out_path).group(1))
        try:
            with open(out_path) as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue

        start_dt    = None
        in_proteins = False

        for line in lines:
            # Header line: "Task 39 | Lines 191–195 of 2441 | Sat Jun  6 07:58:06 PM EDT 2026"
            m = re.match(
                r'Task\s+\d+\s*\|\s*Lines\s+\d+[–\-]+\d+\s+of\s+\d+\s*\|\s*(.+)',
                line
            )
            if m:
                try:
                    start_dt = parse_start_time(m.group(1))
                except ValueError as exc:
                    print(f"  WARNING: could not parse time '{m.group(1)}': {exc}",
                          file=sys.stderr)
                in_proteins = False
                continue

            if line.strip() == 'Proteins in this chunk:':
                in_proteins = True
                continue

            # End of protein block
            if in_proteins and (line.strip() == '' or re.match(r'Task\s+\d+', line)):
                in_proteins = False

            if in_proteins and line.strip():
                pid = line.strip()
                protein_map[pid] = {'task_id': task_id, 'start_dt': start_dt}

    return protein_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Report per-protein topology feature computation time."
    )
    parser.add_argument('--topo_dir', required=True,
                        help="Directory containing <protein_id>.npy output files.")
    parser.add_argument('--log_dir', required=True,
                        help="Directory containing topo_<jobid>_<taskid>.out files.")
    parser.add_argument('--job_id', required=True,
                        help="SLURM array job ID (e.g. 9361260).")
    parser.add_argument('--output', default=None,
                        help="Write TSV to this path (also printed to stdout).")
    parser.add_argument('--sort', default='duration',
                        choices=['protein', 'task_id', 'duration', 'finish_time'],
                        help="Sort order (default: duration, slowest first).")
    args = parser.parse_args()

    # ── Build protein → task/start map from logs ─────────────────────────────
    print(f"Scanning .out logs for job {args.job_id} in {args.log_dir} ...",
          file=sys.stderr)
    protein_map = build_protein_map(args.log_dir, args.job_id)
    print(f"  {len(protein_map)} proteins found across log files.", file=sys.stderr)

    # ── Find completed .npy files ─────────────────────────────────────────────
    npy_files = glob.glob(os.path.join(args.topo_dir, '*.npy'))
    print(f"  {len(npy_files)} .npy files found in {args.topo_dir}.", file=sys.stderr)

    rows        = []
    no_log      = []   # .npy exists but no log entry (different job?)

    for npy_path in npy_files:
        pid       = os.path.splitext(os.path.basename(npy_path))[0]
        finish_dt = datetime.fromtimestamp(os.path.getmtime(npy_path))

        if pid not in protein_map:
            no_log.append(pid)
            continue

        info     = protein_map[pid]
        start_dt = info['start_dt']

        if start_dt is None:
            duration_s   = None
            duration_str = 'unknown_start'
        else:
            duration_s   = (finish_dt - start_dt).total_seconds()
            duration_str = fmt_duration(duration_s)

        rows.append({
            'protein_id':  pid,
            'task_id':     info['task_id'],
            'start_time':  start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else '',
            'finish_time': finish_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'duration':    duration_str,
            'duration_s':  round(duration_s, 1) if duration_s is not None else '',
        })

    if no_log:
        print(f"  WARNING: {len(no_log)} .npy file(s) had no matching log entry "
              f"(may belong to a different job ID). First few: {no_log[:5]}",
              file=sys.stderr)

    # ── Sort ──────────────────────────────────────────────────────────────────
    def sort_key(r):
        if args.sort == 'protein':
            return r['protein_id']
        if args.sort == 'task_id':
            return r['task_id']
        if args.sort == 'finish_time':
            return r['finish_time']
        # 'duration': slowest first; put unknowns at end
        return -(float(r['duration_s']) if r['duration_s'] != '' else -1)

    rows.sort(key=sort_key)

    # ── Summary statistics ────────────────────────────────────────────────────
    valid = [r for r in rows if isinstance(r['duration_s'], float) and r['duration_s'] >= 0]
    print(f"\nCompleted proteins with timing data: {len(valid)}", file=sys.stderr)
    if valid:
        durs = [r['duration_s'] for r in valid]
        print(f"  Min:    {fmt_duration(min(durs))}", file=sys.stderr)
        print(f"  Median: {fmt_duration(sorted(durs)[len(durs)//2])}", file=sys.stderr)
        print(f"  Mean:   {fmt_duration(sum(durs)/len(durs))}", file=sys.stderr)
        print(f"  Max:    {fmt_duration(max(durs))}", file=sys.stderr)
        # Histogram buckets
        buckets = [
            ('<15 min',    lambda s: s < 900),
            ('15–30 min',  lambda s: 900  <= s < 1800),
            ('30–60 min',  lambda s: 1800 <= s < 3600),
            ('1–2 h',      lambda s: 3600 <= s < 7200),
            ('2–4 h',      lambda s: 7200 <= s < 14400),
            ('>4 h',       lambda s: s >= 14400),
        ]
        print("\n  Duration distribution:", file=sys.stderr)
        for label, fn in buckets:
            n = sum(1 for s in durs if fn(s))
            bar = '█' * (n * 30 // len(durs)) if durs else ''
            print(f"    {label:<12}  {n:4d}  {bar}", file=sys.stderr)
    print(file=sys.stderr)

    # ── TSV output ────────────────────────────────────────────────────────────
    headers = ['protein_id', 'task_id', 'start_time', 'finish_time', 'duration', 'duration_s']
    tsv_lines = ['\t'.join(headers)]
    for row in rows:
        tsv_lines.append('\t'.join(str(row[h]) for h in headers))

    output_text = '\n'.join(tsv_lines) + '\n'
    print(output_text)

    if args.output:
        with open(args.output, 'w') as fh:
            fh.write(output_text)
        print(f"Report written to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

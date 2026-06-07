#!/usr/bin/env python3
"""Parse SLURM array job logs for topo feature extraction tasks.

Scans a log directory for topo_<jobid>_<taskid>.out/.err file pairs and
reports status (done/cancelled/running), proteins processed, wall time, and
OK/error counts per task.

Usage
-----
    python parse_topo_logs.py \\
        --log_dir  /mnt/research/woldring_lab/TopoFormer-MF/logs \\
        --job_id   9361260

    # Write TSV report to a file
    python parse_topo_logs.py \\
        --log_dir  /mnt/research/woldring_lab/TopoFormer-MF/logs \\
        --job_id   9361260 \\
        --output   topo_9361260_report.tsv

    # Show only cancelled tasks (to identify proteins that need re-running)
    python parse_topo_logs.py \\
        --log_dir  /mnt/research/woldring_lab/TopoFormer-MF/logs \\
        --job_id   9361260 \\
        --status   cancelled
"""

import argparse
import glob
import os
import re
import sys


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_out_file(path):
    """Parse a .out log file.

    Expected format:
        Task 39 | Lines 191–195 of 2441 | Sat Jun  6 07:58:06 PM EDT 2026
        Node: amr-203 | CPUs: 5
        Proteins in this chunk:
        C0HL62
        ...

        Task 39 done | Sat Jun  6 10:30:24 PM EDT 2026
    """
    result = {
        'task_id':    None,
        'line_start': None,
        'line_end':   None,
        'proteins':   [],
        'start_time': None,
        'end_time':   None,
        'node':       None,
    }
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return result

    in_proteins = False
    for line in lines:
        # Header: "Task 39 | Lines 191–195 of 2441 | Sat Jun  6 07:58:06 PM EDT 2026"
        m = re.match(
            r'Task\s+(\d+)\s*\|\s*Lines\s+(\d+)[–\-]+(\d+)\s+of\s+\d+\s*\|\s*(.+)',
            line
        )
        if m:
            result['task_id']    = int(m.group(1))
            result['line_start'] = int(m.group(2))
            result['line_end']   = int(m.group(3))
            result['start_time'] = m.group(4).strip()
            in_proteins = False
            continue

        # Node line: "Node: amr-203 | CPUs: 5"
        m = re.match(r'Node:\s*(\S+)', line)
        if m:
            result['node'] = m.group(1)
            continue

        # Protein list header
        if line.strip() == 'Proteins in this chunk:':
            in_proteins = True
            continue

        # Done line: "Task 39 done | Sat Jun  6 10:30:24 PM EDT 2026"
        m = re.match(r'Task\s+\d+\s+done\s*\|\s*(.+)', line)
        if m:
            result['end_time'] = m.group(1).strip()
            in_proteins = False
            continue

        # Protein ID lines (stop on blank line)
        if in_proteins:
            if line.strip() == '':
                in_proteins = False
            else:
                result['proteins'].append(line.strip())

    return result


def parse_err_file(path):
    """Parse a .err log file.

    Completed:
        ... INFO Done in 9137.0s — OK: 5, Skipped: 0, Errors: 0

    Cancelled:
        ... error: *** JOB ... CANCELLED AT ... DUE TO TIME LIMIT ***

    Still running:
        ... INFO Mode: ensemble_motion | Processing 5 proteins with 5 workers ...
        (no Done or CANCELLED line yet)
    """
    result = {
        'status':     'unknown',
        'duration_s': None,
        'ok':         None,
        'skipped':    None,
        'errors':     None,
    }
    try:
        with open(path) as fh:
            content = fh.read()
    except OSError:
        return result

    # Successful completion
    m = re.search(
        r'Done in\s+([\d.]+)s\s*[—\-]+\s*OK:\s*(\d+),\s*Skipped:\s*(\d+),\s*Errors:\s*(\d+)',
        content
    )
    if m:
        result['status']     = 'done'
        result['duration_s'] = float(m.group(1))
        result['ok']         = int(m.group(2))
        result['skipped']    = int(m.group(3))
        result['errors']     = int(m.group(4))
        return result

    # Cancelled by SLURM
    if 'CANCELLED' in content and 'TIME LIMIT' in content:
        result['status'] = 'cancelled'
        return result

    # Still running (started but no terminal line yet)
    if 'Mode: ensemble_motion' in content:
        result['status'] = 'running'

    return result


def fmt_duration(seconds):
    if seconds is None:
        return ''
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarise SLURM topo feature extraction task logs."
    )
    parser.add_argument('--log_dir', required=True,
                        help="Directory containing topo_<jobid>_<taskid>.out/err files.")
    parser.add_argument('--job_id', required=True,
                        help="SLURM array job ID (e.g. 9361260).")
    parser.add_argument('--output', default=None,
                        help="Path to write TSV report (default: print to stdout only).")
    parser.add_argument('--status', nargs='+', default=None,
                        choices=['done', 'cancelled', 'running', 'unknown'],
                        help="Only show tasks with these statuses.")
    args = parser.parse_args()

    pattern  = os.path.join(args.log_dir, f'topo_{args.job_id}_*.out')
    out_files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'_(\d+)\.out$', f).group(1))
    )

    if not out_files:
        print(f"ERROR: no files match {pattern}", file=sys.stderr)
        sys.exit(1)

    rows = []
    status_counts = {'done': 0, 'cancelled': 0, 'running': 0, 'unknown': 0}

    for out_path in out_files:
        task_id  = int(re.search(r'_(\d+)\.out$', out_path).group(1))
        err_path = out_path[:-4] + '.err'

        out = parse_out_file(out_path)
        err = parse_err_file(err_path) if os.path.exists(err_path) else {
            'status': 'unknown', 'duration_s': None,
            'ok': None, 'skipped': None, 'errors': None,
        }

        status = err['status']
        status_counts[status] = status_counts.get(status, 0) + 1

        rows.append({
            'task_id':    task_id,
            'status':     status,
            'node':       out.get('node') or '',
            'line_range': (f"{out['line_start']}-{out['line_end']}"
                           if out['line_start'] else ''),
            'n_proteins': len(out['proteins']),
            'proteins':   ','.join(out['proteins']),
            'duration':   fmt_duration(err['duration_s']),
            'duration_s': err['duration_s'] if err['duration_s'] is not None else '',
            'ok':         err['ok']      if err['ok']      is not None else '',
            'skipped':    err['skipped'] if err['skipped'] is not None else '',
            'errors':     err['errors']  if err['errors']  is not None else '',
            'start_time': out.get('start_time') or '',
        })

    # ── Summary block ────────────────────────────────────────────────────────
    total = len(rows)
    print(f"\nJob {args.job_id} | {total} task log files found in {args.log_dir}")
    for s in ('done', 'cancelled', 'running', 'unknown'):
        n   = status_counts[s]
        pct = n / total * 100 if total else 0
        print(f"  {s:<12} {n:4d}  ({pct:.1f}%)")

    done_rows = [r for r in rows if r['status'] == 'done' and r['duration_s'] != '']
    if done_rows:
        durations = [float(r['duration_s']) for r in done_rows]
        print(f"\nWall time for completed tasks (n={len(done_rows)}):")
        print(f"  Min  {fmt_duration(min(durations))}")
        print(f"  Mean {fmt_duration(sum(durations)/len(durations))}")
        print(f"  Max  {fmt_duration(max(durations))}")

    cancelled = [r for r in rows if r['status'] == 'cancelled']
    if cancelled:
        print(f"\nCancelled task IDs (need resubmission):")
        for r in cancelled:
            print(f"  task {r['task_id']:4d} | {r['line_range']:>10} | {r['proteins']}")
    print()

    # ── Apply status filter ──────────────────────────────────────────────────
    if args.status:
        rows = [r for r in rows if r['status'] in args.status]

    # ── TSV output ───────────────────────────────────────────────────────────
    headers = [
        'task_id', 'status', 'node', 'line_range', 'n_proteins',
        'proteins', 'duration', 'ok', 'skipped', 'errors', 'start_time',
    ]

    lines = ['\t'.join(headers)]
    for row in rows:
        lines.append('\t'.join(str(row[h]) for h in headers))

    print('\n'.join(lines))

    if args.output:
        with open(args.output, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"\nReport written to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

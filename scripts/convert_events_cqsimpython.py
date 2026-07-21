#!/usr/bin/env python3
"""convert_events_cqsimpython.py — Convert each driver's events.csv in an
experiment's results into a cqsim-python-comparable events_cqsimpython.csv.

The C++ simulator's events.csv logs a "SchedulingCycle" event and marks
backfilled jobs with "Backfill", neither of which cqsim-python's job-event
log produces (see cqsim-python/src/IOModule/Output_log.py print_event).
To make output diffable against the python simulator, this script:

    1. Drops all "SchedulingCycle" rows.
    2. Renames "Backfill" events to "Run" (both mark a job's start time).

Usage
-----
    python3 scripts/convert_events_cqsimpython.py experiments/exp3a.json

For each driver in the experiment config, reads
    <output_dir>/<driver_tag>/events.csv
and writes
    <output_dir>/<driver_tag>/events_cqsimpython.csv
"""

import argparse
import csv
import json
import os
import sys

DROP_EVENTS = {"SchedulingCycle"}
RENAME_EVENTS = {"Backfill": "Run"}


def convert_events_file(src_path, dst_path):
    with open(src_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            if row["event"] in DROP_EVENTS:
                continue
            row["event"] = RENAME_EVENTS.get(row["event"], row["event"])
            rows.append(row)

    with open(dst_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the experiment JSON config (e.g. experiments/exp3a.json)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    output_dir = config["output_dir"]
    drivers = config.get("drivers", [])
    if not drivers:
        print(f"[error] no drivers found in {args.config}", file=sys.stderr)
        sys.exit(1)

    for driver in drivers:
        tag = driver["tag"]
        src_path = os.path.join(output_dir, tag, "events.csv")
        dst_path = os.path.join(output_dir, tag, "events_cqsimpython.csv")

        if not os.path.exists(src_path):
            print(f"[skip {tag}] missing file: {src_path}", file=sys.stderr)
            continue

        num_rows = convert_events_file(src_path, dst_path)
        print(f"[{tag}] wrote {dst_path} ({num_rows} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared parsing utilities for the cleaned-trace plotting scripts."""

import csv
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta


def load_experiment_config(config_path):
    """Load and return an experiment JSON config."""
    with open(config_path) as f:
        return json.load(f)


def infer_trace_family(config):
    """Infer the trace family from a config dict.

    Returns:
        'theta' or 'polaris'

    Raises:
        ValueError if the config does not clearly target a supported trace.
    """
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")

    hints = []
    machine = config.get("machine")
    if isinstance(machine, str):
        hints.append(machine.lower())

    swf_path = config.get("swf_path")
    if isinstance(swf_path, str):
        hints.append(os.path.basename(swf_path).lower())
        hints.append(swf_path.lower())

    joined = " ".join(hints)
    if "polaris" in joined:
        return "polaris"
    if "theta" in joined:
        return "theta"

    raise ValueError(
        "could not infer plotter from config; expected 'machine' or 'swf_path' "
        "to mention a supported trace such as theta21cln or polaris24cln"
    )


def parse_swf(swf_path):
    """Parse a Standard Workload Format file.

    Returns:
        procs_map     — {job_number: num_requested_processors}
        walltimes_map — {job_number: requested_time_seconds}
        submit_map    — {job_number: unix_submit_time_seconds}
        runtimes_map  — {job_number: actual_run_time_seconds}  (SWF field 4)
    """
    procs_map     = {}
    walltimes_map = {}
    submit_map    = {}
    runtimes_map  = {}
    unix_start    = 0

    with open(swf_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith(';'):
                if 'UnixStartTime:' in line:
                    try:
                        unix_start = int(line.split('UnixStartTime:')[1].strip())
                    except (IndexError, ValueError):
                        pass
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                job_number   = int(parts[0])
                submit_time  = int(parts[1])
                actual_run_t = int(parts[3])   # field 4: actual run time
                num_req_proc = int(parts[7])   # field 8: num_requested_processors
                requested_t  = int(parts[8])   # field 9: requested_time
            except (ValueError, IndexError):
                continue
            procs_map[job_number]     = num_req_proc
            walltimes_map[job_number] = requested_t
            submit_map[job_number]    = unix_start + submit_time
            runtimes_map[job_number]  = actual_run_t

    return procs_map, walltimes_map, submit_map, runtimes_map


# Maintenance event type strings as emitted by event.h type_str()
_MAINT_EVENTS = {
    'SMA', 'SMS', 'SME', 'UMS', 'UME',
    # Legacy names (pre-redesign) kept for backwards compatibility
    'MaintenanceAnnounced', 'MaintenanceStart', 'MaintenanceEnd',
    'ScheduledMaintenanceAnnounced', 'ScheduledMaintenanceStart',
    'ScheduledMaintenanceEnd', 'UnscheduledMaintenanceStart',
    'UnscheduledMaintenanceEnd',
}


def parse_events(events_csv_path):
    """Parse events.csv produced by the simulator.

    Expected named CSV columns: sim_time, event, id (order does not matter).

    Returns:
        submit — {job_id: first_submit_time}
        start  — {job_id: last_run_start_time}   (last Run or Backfill event)
        end    — {job_id: end_time}

    Design notes:
    - Maintenance events (SMA/SMS/SME/UMS/UME) are silently ignored.
    - Resubmit events do NOT update submit time; the original Submit time is
      kept so that wait = start - submit correctly spans any maintenance
      interruption.
    - start[job_id] is OVERWRITTEN on each Run/Backfill event so that
      resubmitted jobs record their final (post-maintenance) run start time.
    """
    submit = {}
    start  = {}
    end    = {}

    with open(events_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row.get('event', '')
            if event in _MAINT_EVENTS:
                continue
            try:
                job_id = int(row['id'])
                t      = float(row['sim_time'])
            except (KeyError, ValueError, TypeError):
                continue

            if event == 'Submit':
                if job_id not in submit:        # keep first (original) submit time
                    submit[job_id] = t
            elif event == 'Resubmit':
                pass                            # re-entry after maintenance; submit unchanged
            elif event in ('Run', 'Backfill'):
                start[job_id] = t               # overwrite → last start time
            elif event == 'End':
                end[job_id] = t

    return submit, start, end


def build_daily_time_series(end_map, tz, value_by_job=None, end_time_limit=None):
    """Aggregate completed work into contiguous per-day buckets.

    Args:
        end_map: {job_id: completion_unix_time_seconds}
        tz: timezone for the day buckets and returned datetimes
        value_by_job: optional {job_id: numeric_value}; defaults to 1 per job
        end_time_limit: optional unix timestamp; completions after this are skipped

    Returns:
        days   — [datetime at local midnight, ...]
        values — [daily aggregate, ...]
    """
    daily = defaultdict(float)
    first_day = None
    last_day = None

    for job_id, raw_end_time in end_map.items():
        try:
            end_time = float(raw_end_time)
        except (TypeError, ValueError):
            continue
        if end_time_limit is not None and end_time > end_time_limit:
            continue

        day = datetime.fromtimestamp(end_time, tz=tz).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        value = 1.0
        if value_by_job is not None:
            try:
                value = float(value_by_job.get(job_id, 0.0))
            except (TypeError, ValueError):
                value = 0.0

        daily[day] += value
        first_day = day if first_day is None or day < first_day else first_day
        last_day = day if last_day is None or day > last_day else last_day

    if first_day is None or last_day is None:
        return [], []

    days = []
    values = []
    day = first_day
    while day <= last_day:
        days.append(day)
        values.append(daily.get(day, 0.0))
        day += timedelta(days=1)

    return days, values


def build_cumulative_completion_series(end_map, tz, end_time_limit=None):
    """Return cumulative completed-job counts over time."""
    completion_times = []
    for raw_end_time in end_map.values():
        try:
            end_time = float(raw_end_time)
        except (TypeError, ValueError):
            continue
        if end_time_limit is not None and end_time > end_time_limit:
            continue
        completion_times.append(end_time)

    completion_times.sort()
    if not completion_times:
        return [], []

    times = []
    cumulative = []
    for idx, end_time in enumerate(completion_times, start=1):
        times.append(datetime.fromtimestamp(end_time, tz=tz))
        cumulative.append(idx)
    return times, cumulative


def rolling_average(values, window):
    """Return a simple trailing rolling average with an expanding warm-up."""
    if window <= 1:
        return [float(v) for v in values]

    averaged = []
    recent = deque()
    running_sum = 0.0
    for value in values:
        numeric = float(value)
        recent.append(numeric)
        running_sum += numeric
        if len(recent) > window:
            running_sum -= recent.popleft()
        averaged.append(running_sum / len(recent))
    return averaged

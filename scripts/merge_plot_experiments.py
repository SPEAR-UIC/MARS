#!/usr/bin/env python3
"""Create a plot-only merged experiment from existing experiment outputs.

The merged config is intended for `scripts/plot_experiment.py`. It does not
launch or rerun any simulations. Instead, it:

1. concatenates the driver lists from multiple experiment JSON files
2. creates a synthetic output directory
3. symlinks each source driver's result directory into that merged output dir

Example:
    python3 scripts/merge_plot_experiments.py experiments/exp8a.json experiments/exp9a.json
    python3 scripts/plot_experiment.py results/plotmerge_exp8a_exp9a/plotmerge.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from plot_results import infer_trace_family, load_experiment_config


REPO_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synthetic plot-only experiment by merging existing experiment outputs."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Two or more experiment JSON configs to merge for plotting.",
    )
    parser.add_argument(
        "--name",
        help="Optional merged name. Defaults to plotmerge_<exp1>_<exp2>...",
    )
    parser.add_argument(
        "--dest",
        help="Optional destination directory. Defaults to results/<name>.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy result directories instead of symlinking them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing merged directory.",
    )
    return parser.parse_args()


def to_abs_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_DIR / path


def ensure_compatible(base_cfg: dict, other_cfg: dict, base_name: str, other_name: str) -> None:
    base_family = infer_trace_family(base_cfg)
    other_family = infer_trace_family(other_cfg)
    if base_family != other_family:
        raise ValueError(
            f"cannot merge {other_name} into {base_name}: trace families differ "
            f"({base_family} vs {other_family})"
        )

    comparable_fields = [
        "swf_path",
        "start_job_index",
        "max_submits",
        "mode",
        "maintenance_csv",
    ]
    mismatches = [
        field
        for field in comparable_fields
        if base_cfg.get(field) != other_cfg.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"cannot merge {other_name} into {base_name}: config mismatch in "
            + ", ".join(mismatches)
        )


def remove_existing(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    if len(args.configs) < 2:
        raise SystemExit("need at least two experiment configs to merge")

    config_paths = [Path(p).resolve() for p in args.configs]
    configs = [load_experiment_config(str(p)) for p in config_paths]
    exp_names = [p.stem for p in config_paths]

    base_cfg = configs[0]
    base_name = exp_names[0]
    for other_cfg, other_name in zip(configs[1:], exp_names[1:]):
        ensure_compatible(base_cfg, other_cfg, base_name, other_name)

    merged_name = args.name or ("plotmerge_" + "_".join(exp_names))
    dest_dir = Path(args.dest).resolve() if args.dest else (REPO_DIR / "results" / merged_name)
    merged_output_dir = dest_dir
    merged_config_path = dest_dir / "plotmerge.json"

    if dest_dir.exists() or dest_dir.is_symlink():
        if not args.force:
            raise SystemExit(f"{dest_dir} already exists; use --force to overwrite it")
        remove_existing(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    merged_cfg = dict(base_cfg)
    merged_cfg["description"] = (
        f"Plot-only merged experiment built from: {', '.join(exp_names)}."
    )
    merged_notes = list(base_cfg.get("notes", []))
    merged_notes.append("This is a generated plot-only merge. Drivers and result links come from multiple experiment configs.")
    merged_notes.append("Sources: " + ", ".join(str(p) for p in config_paths))
    merged_cfg["notes"] = merged_notes
    merged_cfg["output_dir"] = os.path.relpath(merged_output_dir, REPO_DIR)
    merged_cfg["plot_merge_sources"] = [str(p) for p in config_paths]

    merged_drivers = []
    seen_tags: dict[str, str] = {}

    for cfg_path, cfg, exp_name in zip(config_paths, configs, exp_names):
        src_output_dir = to_abs_repo_path(cfg["output_dir"])
        for driver in cfg.get("drivers", []):
            tag = driver.get("tag", driver["type"])
            if tag in seen_tags:
                raise SystemExit(
                    f"duplicate driver tag '{tag}' across {seen_tags[tag]} and {exp_name}; "
                    "rename one of the tags before merging"
                )
            seen_tags[tag] = exp_name
            merged_drivers.append(driver)

            src_driver_dir = src_output_dir / tag
            dest_driver_dir = merged_output_dir / tag
            if not src_driver_dir.exists():
                print(f"[warn] missing results for {exp_name}:{tag} -> {src_driver_dir}")
                continue

            if dest_driver_dir.exists() or dest_driver_dir.is_symlink():
                remove_existing(dest_driver_dir)

            if args.copy:
                shutil.copytree(src_driver_dir, dest_driver_dir)
            else:
                dest_driver_dir.symlink_to(src_driver_dir)

    merged_cfg["drivers"] = merged_drivers

    with open(merged_config_path, "w") as f:
        json.dump(merged_cfg, f, indent=2)
        f.write("\n")

    print(f"Merged config: {merged_config_path}")
    print(f"Merged output: {merged_output_dir}")
    print(f"Drivers: {len(merged_drivers)}")
    print(f"Plot with: python3 scripts/plot_experiment.py {merged_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

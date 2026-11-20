# SchedTwin 2026

CLUSTER26-style plots for the four SchedTwin traces (Frontier 2025, Aurora 2026,
Polaris 2024, Theta 2022). Regenerate with:

```
python3 scripts/plot_schedtwin26.py            # writes SchedTwin26/
```

Inputs are `experiments/schedtwin-*.json` and their runs in `results/schedtwin-*`.

## Differences from CLUSTER26

- Every metric covers only the 10,000 target jobs named in each SWF header;
  warm-up and cool-down jobs are simulated but not counted.
- The traces have no maintenance windows, so `util/util_daily.png` (utilization
  of each 24 h slice of the target window) replaces the uptime and
  drain-before-maintenance plots, and there is no drain resume-time plot.
- Size classes cover every target job (the CLUSTER26 classes would drop about
  half of them):

  | System   | S   | M     | L       | XL       |
  |----------|-----|-------|---------|----------|
  | Frontier | 1   | 2–16  | 17–256  | 257+     |
  | Aurora   | 1   | 2–16  | 17–256  | 257+     |
  | Polaris  | 1   | 2–8   | 9–32    | 33+      |
  | Theta    | 1–8 | 9–128 | 129–512 | 513+     |

- `main_results.csv` holds the paper-table metrics: wt_Mean, wt_Geo
  (exp(mean(ln(1+wait)))), and wt_P99 in seconds with zero waits included, plus
  Util over [first target submit, last target end]. The `wait/` and `bsld/`
  plots and tables follow CLUSTER26 and use only jobs that waited (wait > 0).
- `drain/drain_overview_all_*` is split into two files because that helper draws two systems per figure.

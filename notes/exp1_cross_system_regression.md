# Exp1 Cross-System Regression Analysis

This note compares the `exp1a` and `exp1b` scheduling summaries to answer a simple question:

Do the same heuristic/window choices behave similarly on both systems, or do they diverge a lot?

The analysis script is [analyze_exp1_cross_system.py](/home/cc/CQSimPrivate/scripts/analyze_exp1_cross_system.py:1). It reads the global summary tables from:

- [exp1a wait](/home/cc/CQSimPrivate/results/exp1a_plots/table_wait_overall.csv)
- [exp1a utilization](/home/cc/CQSimPrivate/results/exp1a_plots/table_util_overall.csv)
- [exp1b wait](/home/cc/CQSimPrivate/results/exp1b_plots/table_wait_overall.csv)
- [exp1b utilization](/home/cc/CQSimPrivate/results/exp1b_plots/table_util_overall.csv)

It then matches identical driver tags across both experiments and normalizes each system against its own `FCFS` baseline:

- `wait_reduction = 1 - mean_wait / fcfs_mean_wait`
- `util_delta_pp = utilization_percent - fcfs_utilization_percent`

So positive `wait_reduction` means better than `FCFS`, while negative values mean worse than `FCFS`. `util_delta_pp` is measured in utilization percentage points.

## What I Did

The analysis has two levels.

1. Matched action level.
   I matched every identical driver tag that appears in both experiments, such as `SJF-w64`, `LRF-w16`, and `FCFS`. This gives one paired point per policy-window choice across the two systems.
2. Family mean level.
   I then grouped those matched points by heuristic family, such as `SJF`, `LRF`, `WFP3`, and `F1`, and averaged over window sizes. This gives one point per heuristic family and helps answer whether the broad policy trend transfers, even if some individual windows are noisy.

For each matched point, I computed two normalized cross-system comparison metrics:

- Wait reduction relative to each system's own `FCFS` baseline.
- Utilization change in percentage points relative to each system's own `FCFS` baseline.

Then I fit a simple linear regression of:

- `exp1b metric` as a function of `exp1a metric`

for both wait reduction and utilization delta.

I also report both:

- Pearson correlation, to measure linear agreement.
- Spearman rank correlation, to measure whether the ordering of policies is preserved even if the scale changes.

Finally, I saved:

- point-level comparison tables
- family-level summary tables
- regression summary tables
- two scatter plots with `y = x` and fitted regression lines

## Logic Of The Analysis

The logic is:

- If the same heuristics behave similarly on both systems, then matched points should line up along an increasing trend.
- If they behave almost identically, points should lie close to the `y = x` line.
- If they move in the same direction but with different strength, the trend should still be strong, but the fitted slope will differ from `1`.
- If the systems fundamentally disagree about which heuristics are good or bad, then the correlations should be weak and the scatter should look much more random.

I normalized by each system's own `FCFS` baseline because the absolute scales are not directly comparable.

- Theta and Polaris have different raw wait-time and utilization baselines.
- Using absolute minutes would mix "system size/workload differences" with "heuristic effect differences."
- Using improvement over each system's own `FCFS` isolates the relative effect of the scheduling policy more cleanly.

I used both the matched-action view and the family-mean view on purpose.

- The matched-action view answers: does a specific policy-window choice transfer?
- The family-mean view answers: does the overall heuristic family transfer?

This matters because a family can transfer well overall even if a few windows move around, and conversely a few apparently good matched windows can hide a family that is unstable.

## Outputs

- [all matched actions regression](/home/cc/CQSimPrivate/results/exp1analysis/cross_system_regression_all_points.png)
- [family-mean regression](/home/cc/CQSimPrivate/results/exp1analysis/cross_system_regression_family_means.png)
- [matched action table](/home/cc/CQSimPrivate/results/exp1analysis/matched_policy_points.csv)
- [family summary table](/home/cc/CQSimPrivate/results/exp1analysis/family_summary.csv)
- [regression summary table](/home/cc/CQSimPrivate/results/exp1analysis/regression_summary.csv)

## Regression Summary

| Sample | Metric | N | Slope | R^2 | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| Matched actions | Wait reduction vs FCFS | 160 | 0.618 | 0.780 | 0.883 | 0.891 |
| Matched actions | Utilization delta vs FCFS | 160 | 0.494 | 0.945 | 0.972 | 0.906 |
| Family means | Wait reduction vs FCFS | 16 | 0.703 | 0.919 | 0.959 | 0.935 |
| Family means | Utilization delta vs FCFS | 16 | 0.455 | 0.968 | 0.984 | 0.973 |

## Interpretation

The short answer is: the heuristic behavior is more similar than different, but the effect size is not the same on both systems.

- The direction and ordering transfer well.
  Across all 160 matched non-FCFS actions, wait reduction has strong positive correlation (`r = 0.883`, `rho = 0.891`), and utilization delta is even tighter (`r = 0.972`, `rho = 0.906`).
- The magnitude is compressed on `exp1b` / Polaris.
  The regression slopes are both below `1.0` (`0.618` for wait, `0.494` for utilization), which means a heuristic change that helps or hurts strongly on `exp1a` / Theta tends to have a smaller-magnitude effect on `exp1b` / Polaris.
- At the family level the agreement is even stronger.
  Averaging over windows gives `R^2 = 0.919` for wait and `R^2 = 0.968` for utilization, so the broad heuristic ranking transfers quite well.

In other words, my read is not "the same everywhere" and not "totally different everywhere." It is:

- mostly the same ranking and direction
- noticeably different in response magnitude
- different enough that per-system tuning is still justified

## Where The Systems Differ Most

The biggest wait-transfer gaps from [family summary table](/home/cc/CQSimPrivate/results/exp1analysis/family_summary.csv) are:

- `LRF`: `-58.3%` wait reduction on Theta vs `-29.5%` on Polaris. Harmful on both, but much less harmful on Polaris.
- `LJF`: `-41.3%` vs `-19.0%`. Same pattern.
- `LCF`: `-49.6%` vs `-32.8%`. Also harmful on both, but less so on Polaris.
- `WFP3`: `+23.7%` vs `+14.3%`. Helpful on both, but noticeably weaker on Polaris.
- `FAT`: `+4.0%` vs `-5.3%`. This is one of the clearest family-level sign changes.

The biggest utilization-transfer gaps are:

- `F1`: `-3.09 pp` on Theta vs `-1.61 pp` on Polaris.
- `F3`: `-2.82 pp` vs `-1.49 pp`.
- `SCF`: `-2.82 pp` vs `-1.49 pp`.
- `F2`: `-2.80 pp` vs `-1.55 pp`.
- `F4`: `-2.62 pp` vs `-1.42 pp`.

These are not random differences: they mostly show the same directional story as the regressions. The same policies usually move performance the same way, but Polaris responds less strongly.

## Caveats

This is intentionally a simple regression analysis, so there are a few boundaries to keep in mind.

- It uses only the summary tables, not per-job raw traces.
- It treats matched driver tags as comparable units, which is the right choice for "same heuristic, same window" transfer, but it does not model workload composition directly.
- It uses `FCFS` as the common baseline. That is a sensible normalization for policy comparison, but a different baseline would change the numerical scale.
- It summarizes transfer with linear fits. That is useful for an overall picture, but it will not capture more complicated nonlinear behavior.

So the output should be read as a clear transferability check, not as a full causal model of why the systems differ.

## Bottom Line

The two systems do **not** look wildly different in heuristic behavior.

- If a heuristic/window choice helps on Theta, it is likely to help on Polaris too.
- If it hurts on Theta, it usually hurts on Polaris too.
- The main difference is scale, not direction: Polaris tends to show a smaller response.
- A few families do shift materially enough that cross-system retuning still matters, especially `LRF`, `LJF`, `LCF`, `WFP3`, and `FAT`.

So the most defensible summary is:

The heuristics are broadly transferable across the two systems in ranking and direction, but not identical in magnitude. System-specific tuning still matters.

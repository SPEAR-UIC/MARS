# Single-Column Cross-System Figure Note

This note documents the compact paper-sized version of the cross-system comparison figure:

- [cross_system_regression_all_points.png](/home/cc/CQSimPrivate/results/exp1analysis/cross_system_regression_all_points.png)
- [characteristic_regression_summary.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_regression_summary.csv)
- [characteristic_policy_points.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_policy_points.csv)

## What This Figure Shows

For readability in a single-column paper figure, I did not plot every heuristic family. Instead, I kept a characteristic subset:

- `FCFS`
- `SJF`
- `WFP3`
- `F1`
- `UNICEP`
- `FAT`
- `LRF`
- `LJF`

Each colored cloud still contains all matched window sizes for that family, so the figure is not just one point per family. The open circles mark the family means. `FCFS` is plotted as the black star reference point.

The marker types mean:

- Filled dots: individual matched heuristic-window configurations, such as `SJF-w2`, `SJF-w4`, and `SJF-w8`.
- Hollow rings: the mean of that heuristic family over the displayed window sizes.
- Black star: the `FCFS` reference point.

So the filled dots show within-family variation across windows, while the hollow rings summarize the average position of the family.

The left panel uses wait reduction relative to each system's own `FCFS` baseline:

- `wait_reduction = 1 - mean_wait / fcfs_mean_wait`

So:

- `0` means the same average wait as `FCFS`
- positive values mean lower average wait than `FCFS`
- negative values mean higher average wait than `FCFS`

The right panel uses change in utilization, but with utilization first converted from percent to a 0-to-1 ratio:

- `util_fraction = utilization_percent / 100`
- `delta_util = util_fraction - fcfs_util_fraction`

So:

- `0.00` means the same utilization as `FCFS`
- negative values mean worse utilization than `FCFS`
- positive values mean better utilization than `FCFS`

This keeps the utilization computation in ratio units rather than raw percentages, while still plotting a difference with respect to `FCFS`.

## What R^2 Means Here

`R^2` measures how much of the variation in the Polaris values can be explained by a straight-line relationship with the Theta values.

In this figure:

- Wait panel: `R^2 = 0.72`
- Utilization panel: `R^2 = 0.91`

Interpretation:

- In the wait panel, about 72% of the variation in Polaris wait-response for the displayed heuristic-window points is captured by a linear trend from Theta. That is strong, but not perfect.
- In the utilization panel, about 91% of the variation is captured by the linear trend. That is very strong.

So a higher `R^2` means the two systems are behaving more similarly in magnitude under a simple linear model.

## What Spearman rho Means Here

Spearman `rho` measures rank agreement rather than exact linear agreement. It answers:

If one heuristic-window choice looks better than another on Theta, does it also tend to look better on Polaris?

In this figure:

- Wait panel: `rho = 0.80`
- Utilization panel: `rho = 0.85`

Interpretation:

- `rho = 0.80` for wait means the ordering of the displayed policy-window choices is strongly preserved across systems.
- `rho = 0.85` for utilization means the ordering is preserved even more strongly there.

This is important because two systems can disagree somewhat on the exact effect size while still agreeing on which heuristics are relatively good or bad. Spearman `rho` captures that transfer of ordering.

## How To Read The Two Statistics Together

The two numbers answer different questions:

- `R^2`: do the two systems agree on effect magnitude under a linear relationship?
- `rho`: do the two systems agree on relative ranking?

For this figure, the combined interpretation is:

- wait transfer is strong in ordering and reasonably strong in magnitude
- utilization transfer is very strong in both ordering and magnitude

So the same heuristics are not identical across Theta and Polaris, but they are clearly not random either. The systems broadly agree on which directions are good or bad, and utilization behavior transfers especially cleanly.

## Important Scope Note

These figure-level numbers are for the characteristic subset shown in the compact paper figure, not for the full set of all heuristic families.

This figure is about cross-system transfer of the same heuristic-window action. It does not isolate whether the response to changing the window size itself is preserved across systems. That separate question is documented in:

- [exp1_window_size_preservation.md](/home/cc/CQSimPrivate/notes/exp1_window_size_preservation.md:1)

The broader analysis over all matched policy-window points is still documented in:

- [exp1_cross_system_regression.md](/home/cc/CQSimPrivate/notes/exp1_cross_system_regression.md:1)

That larger note is the comprehensive result. This note is specifically for the paper-sized visualization.

## Paper Figure Description

Figure X compares how a representative subset of scheduling heuristics transfers between the Theta and Polaris systems after normalizing each system against its own FCFS baseline. The left panel plots the change in average wait, while the right panel plots the change in utilization, with each filled dot representing a specific heuristic-window configuration and each hollow circle representing the mean of that heuristic family across its displayed window sizes. Points close to the diagonal indicate similar cross-system behavior, whereas deviations from the diagonal indicate system-specific differences in effect size. The strong positive trends in both panels show that heuristics that help or hurt on Theta tend to do the same on Polaris, although the magnitude of the response is not identical across systems.

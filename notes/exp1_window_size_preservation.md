# Window-Size Preservation Note

This note documents the follow-up analysis that asks a narrower question than the main cross-system figure:

When we keep the heuristic family fixed and only change the window size, does that window-size response transfer from Theta to Polaris?

Relevant outputs:

- [cross_system_window_step_preservation.png](/home/cc/CQSimPrivate/results/exp1analysis/cross_system_window_step_preservation.png)
- [characteristic_window_step_policy_points.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_window_step_policy_points.csv)
- [characteristic_window_step_regression_summary.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_window_step_regression_summary.csv)
- [window_step_policy_points.csv](/home/cc/CQSimPrivate/results/exp1analysis/window_step_policy_points.csv)
- [window_step_regression_summary.csv](/home/cc/CQSimPrivate/results/exp1analysis/window_step_regression_summary.csv)

## What I Did

I started from the same matched exp1a/exp1b table used in the earlier cross-system regression, where each heuristic-window configuration is already normalized against that system's own `FCFS` baseline.

Then, for each heuristic family and for each adjacent window transition, I computed a step response:

- wait step response: the change in `%Δ Avg Wait w.r.t FCFS` when moving from one window to the next, such as `w8 -> w16`
- utilization step response: the change in `Δ Utilization w.r.t FCFS` over that same adjacent window step

So in this figure, a point is no longer a raw scheduling action like `SJF-w32`. Instead, a point is a window transition inside one heuristic family, such as:

- `SJF: w8 -> w16`
- `WFP3: w64 -> w128`

The compact figure uses the same representative family subset as the earlier paper-sized figure, except `FCFS` is excluded because it has no window dimension:

- `SJF`
- `WFP3`
- `F1`
- `UNICEP`
- `FAT`
- `LRF`
- `LJF`

The marker types mean:

- Filled dots: individual adjacent window transitions within a family
- Hollow rings: the mean window-step response of that family across its displayed transitions

## What The Numbers Say

For the compact representative-subset figure:

- Wait step transfer: `R^2 = 0.05`, `rho = 0.20`
- Utilization step transfer: `R^2 = 0.65`, `rho = 0.49`

Interpretation:

- The wait panel is weak. The exact change in wait that results from increasing the window size on Theta does not strongly predict the corresponding change on Polaris.
- The utilization panel is noticeably stronger, but still only moderate compared with the main heuristic-transfer figure.

This is the important distinction:

- heuristic identity is fairly well preserved across systems
- exact window-size tuning is not preserved nearly as well, especially for wait

So if a heuristic family looks broadly good on Theta, it often remains broadly good on Polaris. But that does not mean the best window size, or even the direction and magnitude of every window increase, will transfer cleanly from one system to the other.

## Why This Matters

This result helps separate two ideas that can otherwise get conflated:

1. Heuristic transfer: do the same scheduling ideas help or hurt on both systems?
2. Window-size transfer: does increasing the lookahead window have the same local effect on both systems?

The earlier figure supported the first claim. This new figure shows the second claim is much weaker.

That means it is reasonable to reuse the same heuristic branch set across systems, but it is not safe to assume that one fixed window size will remain optimal across systems or workloads. In other words, the heuristic family transfers more robustly than the exact window parameterization.

## Paper-Ready Description

Figure X evaluates whether the effect of changing the heuristic window size is preserved across Theta and Polaris. Each filled dot represents one adjacent window transition within a heuristic family, such as `w8 -> w16`, after measuring the resulting change relative to each system's own FCFS baseline; each hollow circle shows the mean step response of that heuristic family. The left panel compares the change in normalized average wait, while the right panel compares the change in normalized utilization. Unlike the earlier cross-system heuristic comparison, the wait panel shows only weak agreement across systems, indicating that the effect of increasing window size is not reliably preserved. The utilization panel is more consistent, but still weaker than the direct heuristic-transfer result. This suggests that heuristic families themselves transfer across systems more cleanly than their exact window-size tuning.

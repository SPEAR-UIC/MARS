# Combined Cross-System Transfer Report

This note summarizes the combined 2x2 cross-system transfer figure:

- [cross_system_combined_transfer.png](/home/cc/CQSimPrivate/results/exp1analysis/cross_system_combined_transfer.png)

Supporting tables:

- [characteristic_policy_points.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_policy_points.csv)
- [characteristic_regression_summary.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_regression_summary.csv)
- [characteristic_window_step_policy_points.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_window_step_policy_points.csv)
- [characteristic_window_step_regression_summary.csv](/home/cc/CQSimPrivate/results/exp1analysis/characteristic_window_step_regression_summary.csv)

## What The Four Panels Show

The top row is the direct cross-system action transfer analysis. Each filled dot is one matched heuristic-window action, such as `SJF-w8` or `WFP3-w64`, after normalizing each system against its own `FCFS` baseline. The hollow rings are family means over the displayed window sizes. The black star is the `FCFS` reference point.

- Top-left: `%Δ Avg Wait w.r.t FCFS`, Theta vs Polaris
- Top-right: `Δ Utilization w.r.t FCFS`, Theta vs Polaris

The bottom row is the window-step transfer analysis. Here, each filled dot is an adjacent window transition inside one heuristic family, such as `w8 -> w16`. The hollow rings are the mean step responses for that family.

- Bottom-left: window-step change in `%Δ Avg Wait`
- Bottom-right: window-step change in `Δ Utilization`

## Main Findings

The top row shows that heuristic behavior is broadly preserved across systems.

- Wait transfer: `R^2 = 0.72`, `rho = 0.80`
- Utilization transfer: `R^2 = 0.91`, `rho = 0.85`

This means the same heuristic-window choices that help or hurt on Theta usually help or hurt in the same direction on Polaris. The ordering is also strongly preserved, especially for utilization.

The bottom row shows that exact window-size behavior is much less stable across systems.

- Wait window-step transfer: `R^2 = 0.05`, `rho = 0.20`
- Utilization window-step transfer: `R^2 = 0.65`, `rho = 0.49`

So changing the window from one size to the next does not produce the same local response on both systems, especially for wait. Utilization is more consistent than wait, but still noticeably weaker than the direct heuristic-transfer result.

## Overall Interpretation

The combined result is:

- heuristic identity transfers reasonably well across systems
- exact window-size tuning does not transfer nearly as well

This is the practical conclusion for MCTS-style branching. It is reasonable to keep a shared heuristic branch set because the families themselves show similar cross-system behavior. But it is not safe to assume that one fixed window size, or the same sequence of window-size improvements, will remain optimal across systems or workloads. The heuristic family seems to be the more transferable prior; the exact window choice should remain adaptive.

## Paper-Ready Figure Description

Figure X combines two cross-system transfer analyses for a representative subset of heuristic families. The top row compares the direct behavior of matched heuristic-window actions between Theta and Polaris after normalizing each system against its own FCFS baseline, showing that heuristic effects are largely preserved across systems. The bottom row instead compares the change induced by increasing the window size within each heuristic family, revealing that exact window-size responses transfer much less reliably, especially for average wait. Filled dots denote individual matched actions or adjacent window transitions, while hollow circles denote family means. Together, the panels show that heuristic families transfer across systems more robustly than their precise window-size tuning.

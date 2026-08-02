<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/MARS-Card.png" />
  <source media="(prefers-color-scheme: light)" srcset="docs/images/MARS-Card.png" />
  <img src="docs/images/MARS-Card.png" alt="MARS Card" />
</picture>

---

# MARS: A Monte Carlo Tree Search-based Adaptive and Responsive Scheduler

Modern High Performance Computing (HPC) systems depend on static
heuristics and manual administration for job scheduling and reservation
management. Deep Reinforcement Learning (DRL) has shown promising
scheduling performance but requires historical training data and fixes
the optimization goal at training time, forcing operators to retrain
whenever priorities shift. We introduce **MARS**
(**M**onte Carlo Tree Search-based
**A**daptive and **R**esponsive
**S**cheduler), a training-free HPC scheduler whose
optimization goal is configurable through a reward function rather than
baked into a learned model. MARS uses a lightweight discrete-event
simulator to explore the future consequences of scheduling decisions
within a strict time budget, adapting to the configured reward at each
scheduling cycle. If you use MARS in your work, please cite the following papers:

Yash Kurkure, Yihe Zhang, Zhiling Lan, Michael E. Papka, “MARS: A Monte Carlo Tree Search based Adaptive and Responsive Scheduler”, Proc. of IEEE Cluster, 2026.

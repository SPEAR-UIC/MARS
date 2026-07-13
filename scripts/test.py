import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Your scheduling data
data = {
    'Driver': ['FCFS', 'SJF-w1024', 'F1-w256', 'WFP3-w256', 'MCTS-CW', 'MCTS-CB', 'MCTS-IU', 'MCTS-CU'],
    'P25': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Using 0.1 for log compatibility
    'P50': [55.8500, 19.4500, 21.2333, 56.6667, 11.2333, 22.1167, 53.5333, 149.9333],
    'P75': [537.4542, 298.2833, 149.5833, 379.0250, 100.8417, 277.4500, 419.1333, 671.9375],
    'P85': [828.4908, 501.5133, 292.5600, 597.7117, 417.4050, 698.8700, 914.7733, 1029.5733],
    'P95': [3091.8650, 704.7400, 567.2233, 1085.8900, 1083.6542, 1243.7133, 1612.4033, 1519.4750],
    'P99': [3328.5738, 1304.9207, 870.3640, 1359.4810, 7265.9175, 1972.6547, 2170.4140, 2102.2942],
    'Max': [3346.3500, 1702.3333, 1646.9667, 1688.9000, 7344.7500, 4048.4333, 2460.5833, 2446.9000]
}

df = pd.DataFrame(data)
cols = ['P25', 'P50', 'P75', 'P85', 'P95', 'P99', 'Max']
colors = ['#000000', '#332288', '#117733', '#88CCEE', '#E69F00', '#CC6677', '#AA4499', '#882255']

plt.figure(figsize=(12, 7))

# Plot each driver's line
for i, (index, row) in enumerate(df.iterrows()):
    plt.plot(cols, row[cols], marker='o', color=colors[i], linewidth=2.5, label=row['Driver'], alpha=0.8)

# Formatting for a high-quality research paper
plt.yscale('log')
plt.ylabel('Wait Time (Seconds) - Log Scale', fontsize=14)
plt.xlabel('Wait Time Statistics', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Scheduling Drivers")
plt.tight_layout()

# Save the plot
plt.savefig('parallel_coordinates_wait_times.png', dpi=300)
print("Parallel coordinates plot saved.")
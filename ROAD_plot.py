import matplotlib.pyplot as plt
import numpy as np


avg_scores = [0.174, 0.038, -0.214]
percentiles = [25, 50, 75]
colors = ['yellow', 'blue', 'green']
labels =  ["25th Percentile Average", "50th Percentile Average", "75th Percentile Average"]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(percentiles, avg_scores, color='pink', marker='o', linestyle='-', linewidth=2, markersize=10)

for avg_score, percentile, color, label in zip(avg_scores, percentiles, colors, labels):
    ax.scatter(percentile, avg_score, color=color, s=100, zorder=5, label=f'{label}: {avg_score:.3f}')

ax.set_xlabel('Percentile')
ax.set_ylabel('ROAD Score')
ax.set_title('Percentiles with Highlighted Averages for 2DCNN Model (Accuracy = 85%)')
ax.legend(title='AVG ROAD scores for 50 audio files')

ax.set_ylim(-1, 1)
ax.set_xlim(0, 100)

ax.grid(True, linestyle='--', alpha=0.7)

ax.set_xticks(percentiles)
ax.set_xticklabels([f'{p}th' for p in percentiles])
ax.set_yticks(np.linspace(-1, 1, 9))

plt.tight_layout()
plt.show()

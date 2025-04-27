import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = './result/ablation_result.xlsx'  
df = pd.read_excel(file_path)

labels = df['model'].tolist()
metrics = df.columns.tolist()[1:]
data = df[metrics].values

num_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))

for idx, model_data in enumerate(data):
    values = model_data.tolist()
    values += values[:1]
    ax.plot(angles, values, label=labels[idx], linewidth=2)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
ax.set_rlabel_position(25)
ax.set_ylim(0.6, 0.95)
ax.grid(True)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=10,
    frameon=False
)

plt.tight_layout()

plt.savefig('./result/rader_plot.pdf', bbox_inches='tight', transparent=True)

plt.show()





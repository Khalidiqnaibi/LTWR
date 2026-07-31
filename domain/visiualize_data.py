import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load Data
df_lat = pd.read_csv('eval_results/cve_fusion_latency.csv')
df_met = pd.read_csv('eval_results/cve_metrics_per_query.csv')
df_stat = pd.read_csv('eval_results/cve_stats_report.csv')

# Set visual style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})

# ==========================================
# Figure 1: Fusion Latency Distribution
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
df_lat_long = df_lat.melt(var_name='Fusion Arm', value_name='Latency (microseconds)')
sns.boxplot(
    data=df_lat_long,
    x='Fusion Arm',
    y='Latency (microseconds)',
    ax=ax,
    palette='Set2',
    width=0.4,
    showmeans=True,
    meanprops={
        'marker': 'o',
        'markerfacecolor': 'white',
        'markeredgecolor': 'black',
        'markersize': 8,
    },
)
ax.set_title('CVE Fusion Latency Distribution by Arm')
ax.set_xlabel('Fusion Arm')
ax.set_ylabel('Latency (microseconds)')
plt.tight_layout()
plt.savefig('eval_results/latency_distribution.png', dpi=150)
plt.show()

# ==========================================
# Figure 2: Mean Performance Metrics by Arm
# ==========================================
metrics_to_plot = [
    'real_ndcg3',
    'real_mrr',
    'ndcg3_severity',
    'mrr_authoritative',
]
df_met_long = df_met.melt(
    id_vars=['arm', 'dimension', 'package'],
    value_vars=metrics_to_plot,
    var_name='Metric',
    value_name='Score',
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=df_met_long,
    x='Metric',
    y='Score',
    hue='arm',
    ax=ax,
    palette='Blues_d',
    capsize=0.05,
)
ax.set_title('Mean Performance Metrics by Fusion Arm')
ax.set_xlabel('Metric')
ax.set_ylabel('Mean Score')
ax.legend(title='Arm', loc='upper right')
plt.tight_layout()
plt.savefig('eval_results/metrics_by_arm.png', dpi=150)
plt.show()

# ==========================================
# Figure 3: NDCG@3 Across Query Dimensions
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=df_met,
    x='dimension',
    y='real_ndcg3',
    hue='arm',
    ax=ax,
    palette='Set2',
    capsize=0.05,
)
ax.set_title('real_ndcg3 Score Across Query Dimensions')
ax.set_xlabel('Query Dimension')
ax.set_ylabel('Mean real_ndcg3')
ax.legend(title='Arm', loc='upper right')
plt.tight_layout()
plt.savefig('eval_results/ndcg_by_dimension.png', dpi=150)
plt.show()

# ==========================================
# Figure 4: Statistical Comparison & Mean Delta
# ==========================================
fig, ax = plt.subplots(figsize=(11, 6))
df_stat_sorted = df_stat.sort_values(by='mean_delta', ascending=True)

# Highlight statistically significant differences in green
colors = [
    '#2ca02c' if sig else '#1f77b4' for sig in df_stat_sorted['significant']
]
bars = ax.barh(
    df_stat_sorted['metric'],
    df_stat_sorted['mean_delta'],
    color=colors,
    alpha=0.85,
)
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_title(
    'Statistical Comparison: Mean Delta by Metric & Comparison\n(Green ='
    ' Statistically Significant after Holm correction)'
)
ax.set_xlabel('Mean Delta')
ax.set_ylabel('Comparison Metric')

# Annotate each bar with its mean delta value and significance marker
for bar, sig in zip(bars, df_stat_sorted['significant']):
  val = bar.get_width()
  label = f'{val:+.3f}' + (' *' if sig else '')
  offset = 0.008 if val >= 0 else -0.008
  ha = 'left' if val >= 0 else 'right'
  ax.text(
      val + offset,
      bar.get_y() + bar.get_height() / 2,
      label,
      va='center',
      ha=ha,
      fontsize=9,
  )

# Provide margin for text labels
xlim = ax.get_xlim()
ax.set_xlim(xlim[0] - 0.05, xlim[1] + 0.06)

plt.tight_layout()
plt.savefig('eval_results/mean_delta.png', dpi=150)
plt.show()
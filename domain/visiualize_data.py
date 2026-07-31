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

# Mapping dictionaries for clean display labels
ARM_MAP = {'rrf': 'RRF', 'static_twr': 'Static TWR', 'ltwr': 'LTWR'}

METRIC_MAP = {
    'real_ndcg3': 'NDCG@3 (Overall)',
    'real_mrr': 'MRR (Overall)',
    'ndcg3_severity': 'NDCG@3 (Severity)',
    'mrr_authoritative': 'MRR (Authoritative)',
}

DIM_MAP = {
    'severity': 'Severity',
    'vuln_status': 'Vulnerability Status',
    'recency': 'Recency',
    'combined': 'Combined',
}

# Standardized palette for arms across all plots
PALETTE = 'Set2'

# ==========================================
# Figure 1: Fusion Latency Distribution
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
df_lat_rename = df_lat.rename(columns=ARM_MAP)
df_lat_long = df_lat_rename.melt(
    var_name='Fusion Arm', value_name='Latency (μs)'
)

sns.boxplot(
    data=df_lat_long,
    x='Fusion Arm',
    y='Latency (μs)',
    ax=ax,
    palette=PALETTE,
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
ax.set_ylabel('Latency (μs)')

plt.tight_layout()
plt.savefig('eval_results/latency_distribution.png', dpi=150)
plt.show()

# ==========================================
# Figure 2: Mean Performance Metrics by Arm
# ==========================================
metrics_to_plot = list(METRIC_MAP.keys())

df_met_clean = df_met.copy()
df_met_clean['arm_display'] = df_met_clean['arm'].map(ARM_MAP)

df_met_long = df_met_clean.melt(
    id_vars=['arm_display', 'dimension', 'package'],
    value_vars=metrics_to_plot,
    var_name='Metric',
    value_name='Score',
)
df_met_long['Metric_Display'] = df_met_long['Metric'].map(METRIC_MAP)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=df_met_long,
    x='Metric_Display',
    y='Score',
    hue='arm_display',
    ax=ax,
    palette=PALETTE,
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
df_met_clean['dim_display'] = df_met_clean['dimension'].map(DIM_MAP)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=df_met_clean,
    x='dim_display',
    y='real_ndcg3',
    hue='arm_display',
    ax=ax,
    palette=PALETTE,
    capsize=0.05,
)
ax.set_title('NDCG@3 Score Across Query Dimensions')
ax.set_xlabel('Query Dimension')
ax.set_ylabel('Mean NDCG@3')

# Place legend outside to avoid obscuring bars
ax.legend(title='Arm', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig('eval_results/ndcg_by_dimension.png', dpi=150)
plt.show()

# ==========================================
# Figure 4: Statistical Comparison & Mean Delta
# ==========================================
fig, ax = plt.subplots(figsize=(11, 6))

# Clean up raw comparison string labels (e.g., real_ndcg3::static_twr_vs_rrf)
df_stat_sorted = df_stat.sort_values(by='mean_delta', ascending=True).copy()


def format_stat_label(label_str):
  if '::' in label_str:
    metric, comp = label_str.split('::')
    clean_metric = METRIC_MAP.get(metric, metric)
    clean_comp = comp.replace('_vs_', ' vs. ').replace('_', ' ').upper()
    return f'{clean_metric} ({clean_comp})'
  return label_str


df_stat_sorted['clean_label'] = df_stat_sorted['metric'].apply(
    format_stat_label
)

colors = [
    '#2ca02c' if sig else '#1f77b4' for sig in df_stat_sorted['significant']
]
bars = ax.barh(
    df_stat_sorted['clean_label'],
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
ax.set_ylabel('Comparison & Metric')

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

xlim = ax.get_xlim()
ax.set_xlim(xlim[0] - 0.05, xlim[1] + 0.06)

plt.tight_layout()
plt.savefig('eval_results/mean_delta.png', dpi=150)
plt.show()
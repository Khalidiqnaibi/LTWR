import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def VisualizeData(
    lat_path="eval_results/cve_fusion_latency.csv",
    met_path="eval_results/cve_metrics_per_query.csv",
    stat_path="eval_results/cve_stats_report.csv",
):
    # 1. Q1 Academic Formatting Parameters
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.2,
    })
    sns.set_theme(style="whitegrid", rc=plt.rcParams)

    # 2. Load Data
    df_lat = pd.read_csv(lat_path)
    df_met = pd.read_csv(met_path)
    df_stat = pd.read_csv(stat_path)

    # Label Mappings
    ARM_MAP = {"rrf": "RRF", "static_twr": "Static TWR", "ltwr": "LTWR"}
    METRIC_MAP = {
        "real_ndcg3": "NDCG@3 (Overall)",
        "real_mrr": "MRR (Overall)",
        "ndcg3_severity": "NDCG@3 (Severity)",
        "mrr_authoritative": "MRR (Authoritative)",
    }
    DIM_MAP = {
        "severity": "Severity",
        "vuln_status": "Vulnerability Status",
        "recency": "Recency",
        "combined": "Combined",
    }

    # Okabe-Ito inspired colorblind-safe palette for print/digital
    PALETTE = ["#4B8BBE", "#D95F02", "#1B9E77"]

    # ==========================================
    # Figure 1: Fusion Latency Distribution
    # ==========================================
    fig, ax = plt.subplots(figsize=(3.5, 3.0))  # Single-column width
    
    df_lat_rename = df_lat.rename(columns=ARM_MAP)
    arms = list(ARM_MAP.values())
    data_to_plot = [df_lat_rename[arm].dropna() for arm in arms if arm in df_lat_rename]

    bp = ax.boxplot(
        data_to_plot,
        labels=[arm for arm in arms if arm in df_lat_rename],
        patch_artist=True,
        widths=0.4,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="black", linewidth=1),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    # Apply custom colors cleanly
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.6)

    ax.set_title("CVE Fusion Latency Distribution")
    ax.set_xlabel("Fusion Arm")
    ax.set_ylabel("Latency (μs)")
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("eval_results/latency_distribution.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ==========================================
    # Figure 2: Mean Performance Metrics by Arm
    # ==========================================
    metrics_to_plot = list(METRIC_MAP.keys())
    df_met_clean = df_met.copy()
    df_met_clean["arm_display"] = df_met_clean["arm"].map(ARM_MAP)

    df_met_long = df_met_clean.melt(
        id_vars=["arm_display", "dimension", "package"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )
    df_met_long["Metric_Display"] = df_met_long["Metric"].map(METRIC_MAP)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # Double-column width
    sns.barplot(
        data=df_met_long,
        x="Metric_Display",
        y="Score",
        hue="arm_display",
        ax=ax,
        palette=PALETTE,
        edgecolor="black",
        linewidth=0.6,
        capsize=0.06,
        err_kws={"linewidth": 0.8, "color": "black"},
    )
    ax.set_title("Mean Performance Metrics Across Fusion Arms")
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Mean Score")
    ax.legend(title="Arm", loc="upper right", framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("eval_results/metrics_by_arm.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ==========================================
    # Figure 3: NDCG@3 Across Query Dimensions
    # ==========================================
    df_met_clean["dim_display"] = df_met_clean["dimension"].map(DIM_MAP)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sns.barplot(
        data=df_met_clean,
        x="dim_display",
        y="real_ndcg3",
        hue="arm_display",
        ax=ax,
        palette=PALETTE,
        edgecolor="black",
        linewidth=0.6,
        capsize=0.06,
        err_kws={"linewidth": 0.8, "color": "black"},
    )
    ax.set_title("NDCG@3 Score Across Query Dimensions")
    ax.set_xlabel("Query Dimension")
    ax.set_ylabel("Mean NDCG@3")
    ax.legend(title="Arm", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("eval_results/ndcg_by_dimension.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ==========================================
    # Figure 4: Statistical Comparison & Mean Delta
    # ==========================================
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    df_stat_sorted = df_stat.sort_values(by="mean_delta", ascending=True).copy()

    def format_stat_label(label_str):
        if "::" in label_str:
            metric, comp = label_str.split("::")
            clean_metric = METRIC_MAP.get(metric, metric)
            clean_comp = comp.replace("_vs_", " vs. ").replace("_", " ").upper()
            return f"{clean_metric} ({clean_comp})"
        return label_str

    df_stat_sorted["clean_label"] = df_stat_sorted["metric"].apply(format_stat_label)

    # Muted forest green for significant, dark steel grey for non-significant
    colors = ["#1B9E77" if sig else "#7F8C8D" for sig in df_stat_sorted["significant"]]
    bars = ax.barh(
        df_stat_sorted["clean_label"],
        df_stat_sorted["mean_delta"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Statistical Comparison: Mean Delta (Green = Holm-Corrected p < 0.05)")
    ax.set_xlabel("Mean Delta")
    ax.set_ylabel("Comparison & Metric")
    sns.despine(ax=ax)

    for bar, sig in zip(bars, df_stat_sorted["significant"]):
        val = bar.get_width()
        label = f"{val:+.3f}" + (" *" if sig else "")
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, fontsize=8.5)

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.04, xlim[1] + 0.06)

    plt.tight_layout()
    plt.savefig("eval_results/mean_delta.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  [OK] All CVE Figures exported to PDF in eval_results/")


if __name__ == "__main__":
    VisualizeData("results/ltwr/cve_fusion_latency.csv", "results/ltwr/cve_metrics_per_query.csv", "results/ltwr/cve_stats_report.csv")
"""
EduPro Exploratory Data Analysis & Modeling Pipeline
Generates diagnostic figures and quantitative validation charts for reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from src.data_loader import load_raw_data
from src.feature_engineering import engineer_learner_features, get_clustering_feature_matrix
from src.clustering import evaluate_cluster_range, compute_hierarchical_linkage
from scipy.cluster.hierarchy import dendrogram

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    print("Running exploratory data analysis and figure generation...")
    courses_df, users_df, txns_df, teachers_df = load_raw_data()
    features_df = engineer_learner_features()

    # 1. Course Category & Level Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(
        data=courses_df,
        y="CourseCategory",
        order=courses_df["CourseCategory"].value_counts().index,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Catalog Offerings by Subject Domain", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Courses")

    sns.countplot(
        data=courses_df,
        x="CourseLevel",
        hue="CourseType",
        palette="mako",
        ax=axes[1]
    )
    axes[1].set_title("Course Level & Pricing Breakdown", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_course_distribution.png")
    plt.close()

    # 2. Optimal Clusters: Elbow & Silhouette Plot
    X, feat_cols = get_clustering_feature_matrix(features_df)
    eval_df = evaluate_cluster_range(X, k_min=2, k_max=8)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia (WCSS)", color=color, fontweight="bold")
    ax1.plot(eval_df["k"], eval_df["inertia"], marker="o", color=color, linewidth=2, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Silhouette Score", color=color, fontweight="bold")
    ax2.plot(eval_df["k"], eval_df["silhouette_score"], marker="s", color=color, linewidth=2, linestyle="--", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Elbow Method & Silhouette Analysis for Optimal k", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_cluster_evaluation.png")
    plt.close()

    # 3. Hierarchical Clustering Dendrogram
    linkage_matrix, c_coeff = compute_hierarchical_linkage(X)
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
    plt.title(f"Agglomerative Dendrogram (Ward Linkage, Cophenetic r = {c_coeff:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Cluster Group / Sub-sample Index")
    plt.ylabel("Ward Euclidean Distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_hierarchical_dendrogram.png")
    plt.close()

    print(f"All analytical figures generated and saved to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()

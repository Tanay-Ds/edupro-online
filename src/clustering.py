import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

from .config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    CLUSTERED_USERS_FILE,
    CLUSTER_SUMMARY_FILE,
    RANDOM_STATE,
    DEFAULT_N_CLUSTERS,
    MAX_CLUSTERS_EVAL
)
from .feature_engineering import engineer_learner_features, get_clustering_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_cluster_range(X: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """
    Evaluates clustering validity metrics across a range of k values to determine the optimal number of clusters.
    Computes Inertia (WCSS), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        inertia = km.inertia_
        
        results.append({
            "k": k,
            "inertia": float(inertia),
            "silhouette_score": float(sil),
            "davies_bouldin_index": float(db),
            "calinski_harabasz_score": float(ch)
        })

    eval_df = pd.DataFrame(results)
    logger.info("Cluster evaluation summary:\n" + str(eval_df))
    return eval_df


def generate_persona_profiles(clustered_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Synthesizes meaningful persona titles, behavioural descriptions, and recommended strategy
    for each cluster based on centroid characteristics.
    """
    personas = {}
    grouped = clustered_df.groupby("cluster_label")

    for cluster_id, grp in grouped:
        avg_spend = grp["total_spending"].mean()
        avg_courses = grp["total_courses_enrolled"].mean()
        paid_ratio = grp["paid_course_ratio"].mean()
        avg_rating = grp["avg_course_rating_enrolled"].mean()
        learning_depth = grp["learning_depth_index"].mean()
        diversity = grp["diversity_score"].mean()
        top_cat = grp["preferred_category"].mode().iloc[0] if not grp["preferred_category"].empty else "General"
        top_lvl = grp["preferred_level"].mode().iloc[0] if not grp["preferred_level"].empty else "Beginner"
        avg_age = grp["Age"].mean()
        cluster_size = len(grp)
        cluster_share = float(cluster_size / len(clustered_df))

        # Persona logic based on behavioural signatures
        if avg_courses > 8 or avg_spend > 600:
            title = "Career Up-skillers & Power Learners"
            description = (
                "Highly active power learners with extensive enrollment volume and substantial investment. "
                "They explore multi-domain roadmaps, seeking end-to-end skill progression."
            )
            strategy = "Offer enterprise learning subscriptions, executive mentorship, and career transition pathways."
            icon = "💼"
        elif learning_depth > 1.3 or top_lvl == "Advanced":
            title = "Advanced Domain Specialists"
            description = (
                "Experienced practitioners focused on advanced and rigorous subject matters. "
                "They prioritize specialized mastery, high learning depth, and challenging certifications."
            )
            strategy = "Recommend advanced masterclasses, hands-on capstone projects, and peer code reviews."
            icon = "🎯"
        elif top_lvl == "Intermediate" or (diversity > 0.2 and learning_depth >= 0.8):
            title = "Intermediate Skill Builders"
            description = (
                "Mid-tier learners moving beyond basics toward specialized vocational application. "
                "They focus on applied workflows, project certifications, and domain transition."
            )
            strategy = "Deliver intermediate certification tracks, portfolio-building projects, and industry case studies."
            icon = "⚡"
        else:
            title = "Foundational & Budget Learners"
            description = (
                "Early-stage learners focusing heavily on free introductory courses and foundational basics. "
                "Price-sensitive with selective enrollment patterns."
            )
            strategy = "Recommend high-rated free beginner courses and transitional discounts for entry-level paid tracks."
            icon = "🌱"

        personas[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "title": title,
            "icon": icon,
            "description": description,
            "strategy": strategy,
            "size": cluster_size,
            "share_pct": round(cluster_share * 100, 2),
            "metrics": {
                "avg_age": round(float(avg_age), 1),
                "avg_courses_enrolled": round(float(avg_courses), 2),
                "avg_total_spending": round(float(avg_spend), 2),
                "paid_course_ratio": round(float(paid_ratio), 3),
                "avg_course_rating": round(float(avg_rating), 2),
                "learning_depth_index": round(float(learning_depth), 2),
                "diversity_score": round(float(diversity), 3),
                "dominant_category": top_cat,
                "dominant_level": top_lvl
            }
        }

    return personas


def train_clustering_pipeline(n_clusters: int = DEFAULT_N_CLUSTERS) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes full clustering pipeline:
    1. Feature scaling
    2. K-Means clustering model fit
    3. PCA projection for 2D/3D visualization
    4. Model artifact serialization
    5. Generation and saving of cluster profiles
    """
    logger.info("Executing clustering pipeline...")
    features_df = engineer_learner_features()
    X, feature_cols = get_clustering_feature_matrix(features_df)

    # 1. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. KMeans Modeling
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    features_df["cluster_label"] = cluster_labels

    # 3. PCA for Visualization Projections
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca_2d.fit_transform(X_scaled)
    features_df["pca_x"] = coords_2d[:, 0]
    features_df["pca_y"] = coords_2d[:, 1]

    pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
    coords_3d = pca_3d.fit_transform(X_scaled)
    features_df["pca_z"] = coords_3d[:, 2]

    # 4. Generate Cluster Personas
    personas = generate_persona_profiles(features_df, feature_cols)
    features_df["persona_title"] = features_df["cluster_label"].astype(str).map(
        lambda cid: personas.get(cid, {}).get("title", f"Cluster {cid}")
    )

    # 5. Save Artifacts
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(pca_2d, MODELS_DIR / "pca_2d.joblib")
    joblib.dump(pca_3d, MODELS_DIR / "pca_3d.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_column_names.joblib")

    features_df.to_csv(CLUSTERED_USERS_FILE, index=False)
    with open(CLUSTER_SUMMARY_FILE, "w") as f:
        json.dump(personas, f, indent=2)

    logger.info(f"Clustering complete. Models saved to {MODELS_DIR}, dataset saved to {CLUSTERED_USERS_FILE}")
    return features_df, personas


def compute_hierarchical_linkage(X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Computes hierarchical clustering linkage matrix and cophenetic correlation coefficient
    for structural clustering validation.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Subsample if dataset is large for computational efficiency and clean dendrogram
    if len(X_scaled) > 500:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(len(X_scaled), 500, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    linkage_matrix = linkage(X_sample, method="ward", metric="euclidean")
    c_coeff, _ = cophenet(linkage_matrix, pdist(X_sample))
    logger.info(f"Hierarchical Cophenetic Correlation Coefficient: {c_coeff:.4f}")

    return linkage_matrix, float(c_coeff)


if __name__ == "__main__":
    df, personas = train_clustering_pipeline()
    print("Clusters trained successfully.")
    print(json.dumps(personas, indent=2))

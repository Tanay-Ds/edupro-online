"""
EduPro Pipeline Orchestrator
Executes the end-to-end data pipeline:
1. Raw Data Ingestion & Validation
2. Multidimensional Feature Engineering
3. Optimal Hyperparameter Evaluation & Clustering
4. Persona Profiling & Model Serialization
5. Recommendation Matrix Construction & Performance Validation
"""

import sys
import logging
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_raw_data
from src.feature_engineering import engineer_learner_features, get_clustering_feature_matrix
from src.clustering import train_clustering_pipeline, evaluate_cluster_range
from src.recommender import HybridCourseRecommender
from src.evaluation import evaluate_segmentation_quality, evaluate_recommendation_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EduPro-Pipeline")


def main():
    logger.info("=========================================================")
    logger.info("  EDUPRO LEARNER PERSONALIZATION ENGINE: PIPELINE START  ")
    logger.info("=========================================================")

    # Step 1: Ingestion
    logger.info("[Step 1/5] Ingesting raw datasets...")
    courses, users, txns, teachers = load_raw_data()
    logger.info(f"Loaded: {len(users)} users, {len(courses)} courses, {len(txns)} transactions, {len(teachers)} teachers.")

    # Step 2: Feature Engineering
    logger.info("[Step 2/5] Engineering multidimensional learner feature profiles...")
    features_df = engineer_learner_features()
    logger.info(f"Engineered profiles for {len(features_df)} learners across {features_df.shape[1]} features.")

    # Step 3: Clustering & Segmentation
    logger.info("[Step 3/5] Fitting unsupervised clustering model and generating personas...")
    clustered_df, personas = train_clustering_pipeline(n_clusters=4)
    for cid, p in personas.items():
        logger.info(f"  Cluster {cid}: '{p['title']}' ({p['share_pct']}% share, {p['size']} users)")

    # Step 4: Recommender Engine Fitting
    logger.info("[Step 4/5] Initializing cluster-aware hybrid recommendation engine...")
    recommender = HybridCourseRecommender()
    
    # Step 5: System Evaluation
    logger.info("[Step 5/5] Running model validation and offline evaluation benchmarks...")
    X, feat_cols = get_clustering_feature_matrix(clustered_df)
    seg_metrics = evaluate_segmentation_quality(clustered_df, feat_cols)
    rec_metrics = evaluate_recommendation_performance(recommender, sample_size=100)

    logger.info("---------------------------------------------------------")
    logger.info("             SYSTEM PERFORMANCE BENCHMARKS               ")
    logger.info("---------------------------------------------------------")
    logger.info(f"  • Silhouette Score:              {seg_metrics['silhouette_score']}")
    logger.info(f"  • Davies-Bouldin Index:          {seg_metrics['davies_bouldin_index']}")
    logger.info(f"  • Catalog Coverage:              {rec_metrics['catalog_coverage_pct']}%")
    logger.info(f"  • Mean Recommended Rating:       {rec_metrics['mean_recommended_course_rating']} / 5.0")
    logger.info(f"  • Projected Engagement Lift:    +{rec_metrics['projected_engagement_lift_pct']}%")
    logger.info("=========================================================")
    logger.info("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!             ")
    logger.info("  Launch Dashboard via: streamlit run app/main.py        ")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()

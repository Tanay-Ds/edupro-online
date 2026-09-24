import logging
import numpy as np
import pandas as pd
from typing import Tuple, List
from .data_loader import load_raw_data
from .config import PROCESSED_USER_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_category_entropy(cat_series: pd.Series) -> float:
    """
    Computes Shannon entropy over a series of categorical labels as a measure of topical exploration diversity.
    """
    counts = cat_series.value_counts()
    probs = counts / len(cat_series)
    # Shannon entropy base 2
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    return float(entropy)


def engineer_learner_features() -> pd.DataFrame:
    """
    Aggregates user, transaction, course, and teacher records into a multi-dimensional
    learner profile matrix containing engagement, preference, and behavioral metrics.
    """
    courses_df, users_df, txns_df, teachers_df = load_raw_data()
    
    # Merge transaction records with course details
    tx_course = txns_df.merge(
        courses_df[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"]],
        on="CourseID",
        how="left"
    )

    all_categories = sorted(courses_df["CourseCategory"].dropna().unique())
    all_levels = ["Beginner", "Intermediate", "Advanced"]
    total_platform_categories = len(all_categories)

    learner_records = []

    # Group transactions by user
    grouped = tx_course.groupby("UserID")

    for user_id, group in grouped:
        n_courses = len(group)
        dates = group["TransactionDate"].dropna().sort_values()
        
        # 1. Engagement metrics
        if len(dates) > 1:
            active_span = (dates.iloc[-1] - dates.iloc[0]).days
            enrollment_freq = active_span / max(n_courses - 1, 1)
        else:
            active_span = 0
            enrollment_freq = 0.0

        n_unique_categories = group["CourseCategory"].nunique()
        avg_courses_per_cat = n_courses / max(n_unique_categories, 1)
        avg_duration = group["CourseDuration"].mean()

        # 2. Preference metrics
        cat_counts = group["CourseCategory"].value_counts()
        preferred_cat = cat_counts.index[0] if len(cat_counts) > 0 else "Unknown"
        
        level_counts = group["CourseLevel"].value_counts()
        preferred_level = level_counts.index[0] if len(level_counts) > 0 else "Unknown"
        
        avg_rating = group["CourseRating"].mean()
        paid_count = (group["CourseType"].str.lower() == "paid").sum()
        paid_ratio = paid_count / max(n_courses, 1)

        # Level ratios
        n_beg = level_counts.get("Beginner", 0)
        n_int = level_counts.get("Intermediate", 0)
        n_adv = level_counts.get("Advanced", 0)
        
        prop_beginner = n_beg / n_courses
        prop_intermediate = n_int / n_courses
        prop_advanced = n_adv / n_courses

        # Category proportions
        cat_props = {}
        for cat in all_categories:
            cat_props[f"cat_prop_{cat.replace(' ', '_')}"] = cat_counts.get(cat, 0) / n_courses

        # 3. Behavioral metrics
        total_spend = group["Amount"].sum()
        avg_spend = total_spend / max(n_courses, 1)
        diversity_score = n_unique_categories / max(total_platform_categories, 1)
        category_entropy = compute_category_entropy(group["CourseCategory"])
        
        # Learning depth: ratio of advanced & intermediate experience over beginner baseline
        learning_depth = (n_int + 2 * n_adv) / max(n_courses, 1)

        rec = {
            "UserID": user_id,
            "total_courses_enrolled": n_courses,
            "unique_categories_count": n_unique_categories,
            "avg_courses_per_category": avg_courses_per_cat,
            "active_span_days": active_span,
            "enrollment_frequency_days": enrollment_freq,
            "avg_course_duration": avg_duration,
            "preferred_category": preferred_cat,
            "preferred_level": preferred_level,
            "avg_course_rating_enrolled": avg_rating,
            "paid_course_ratio": paid_ratio,
            "prop_beginner": prop_beginner,
            "prop_intermediate": prop_intermediate,
            "prop_advanced": prop_advanced,
            "total_spending": total_spend,
            "avg_spending_per_course": avg_spend,
            "diversity_score": diversity_score,
            "category_entropy": category_entropy,
            "learning_depth_index": learning_depth,
            **cat_props
        }
        learner_records.append(rec)

    features_df = pd.DataFrame(learner_records)

    # Merge demographics from Users table
    final_df = users_df.merge(features_df, on="UserID", how="inner")
    
    # Encode gender binary for ML (Male=1, Female=0, Unknown=0.5)
    gender_map = {"Male": 1.0, "Female": 0.0}
    final_df["gender_binary"] = final_df["Gender"].map(gender_map).fillna(0.5)

    logger.info(f"Learner features engineered for {len(final_df)} active platform users.")
    
    # Cache processed dataset
    final_df.to_parquet(PROCESSED_USER_FEATURES, index=False)
    logger.info(f"Saved feature dataset to {PROCESSED_USER_FEATURES}")

    return final_df


def get_clustering_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts the numerical feature matrix specifically designed for learner clustering.
    Excludes non-behavioral identifiers like UserID, UserName, Email.
    """
    feature_cols = [
        "Age",
        "gender_binary",
        "total_courses_enrolled",
        "avg_courses_per_category",
        "active_span_days",
        "enrollment_frequency_days",
        "avg_course_duration",
        "avg_course_rating_enrolled",
        "paid_course_ratio",
        "prop_beginner",
        "prop_intermediate",
        "prop_advanced",
        "total_spending",
        "avg_spending_per_course",
        "diversity_score",
        "category_entropy",
        "learning_depth_index"
    ]
    
    # Add category proportion columns
    cat_prop_cols = [c for c in df.columns if c.startswith("cat_prop_")]
    all_features = feature_cols + cat_prop_cols

    return df[all_features].copy(), all_features


if __name__ == "__main__":
    df = engineer_learner_features()
    X, cols = get_clustering_feature_matrix(df)
    print(f"Matrix shape: {X.shape}, Features count: {len(cols)}")

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .config import CLUSTERED_USERS_FILE, COURSES_FILE, TRANSACTIONS_FILE, TEACHERS_FILE
from .data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HybridCourseRecommender:
    """
    Production-grade cluster-aware hybrid recommendation engine combining:
    1. Content-based course vector similarity
    2. Learner cluster collaborative affinity
    3. Quality and rating weighting
    4. Real-time filtering and personalized explainability
    """

    def __init__(self, content_weight: float = 0.40, cluster_weight: float = 0.40, rating_weight: float = 0.20):
        self.w_cb = content_weight
        self.w_cl = cluster_weight
        self.w_rt = rating_weight
        
        self.courses_df: pd.DataFrame = None
        self.users_df: pd.DataFrame = None
        self.txns_df: pd.DataFrame = None
        self.teachers_df: pd.DataFrame = None
        self.clustered_users_df: pd.DataFrame = None
        
        self.course_feature_matrix: np.ndarray = None
        self.course_id_to_idx: Dict[str, int] = {}
        self.idx_to_course_id: Dict[int, str] = {}
        self.cluster_course_affinity: Dict[int, Dict[str, float]] = {}

        self._fit_pipeline()

    def _fit_pipeline(self):
        """
        Loads data, constructs course feature representations, and builds cluster affinity matrices.
        """
        self.courses_df, self.users_df, self.txns_df, self.teachers_df = load_raw_data()
        
        # Load clustered users if available, otherwise aggregate
        try:
            self.clustered_users_df = pd.read_csv(CLUSTERED_USERS_FILE)
        except Exception:
            logger.warning("Clustered users file not found. Fallback to basic user dataset.")
            self.clustered_users_df = self.users_df.copy()
            self.clustered_users_df["cluster_label"] = 0

        self._build_course_feature_matrix()
        self._build_cluster_affinity_matrix()
        logger.info("Hybrid Course Recommender initialized successfully.")

    def _build_course_feature_matrix(self):
        """
        Constructs normalized feature embeddings for all catalog courses.
        """
        df = self.courses_df.copy()
        
        # Merge teacher rating if available
        # Note: Teachers table has 60 teachers, matching courses
        df["TeacherRating"] = self.teachers_df["TeacherRating"].values[:len(df)]
        
        # One-hot encode category
        cat_dummies = pd.get_dummies(df["CourseCategory"], prefix="cat", dtype=float)
        
        # One-hot encode level
        lvl_dummies = pd.get_dummies(df["CourseLevel"], prefix="lvl", dtype=float)
        
        # Encode course type (Paid=1, Free=0)
        is_paid = (df["CourseType"].str.lower() == "paid").astype(float).values.reshape(-1, 1)

        # Scale numericals
        scaler = MinMaxScaler()
        num_cols = ["CoursePrice", "CourseDuration", "CourseRating", "TeacherRating"]
        scaled_nums = scaler.fit_transform(df[num_cols].fillna(0.0))

        feature_matrix = np.hstack([cat_dummies.values, lvl_dummies.values, is_paid, scaled_nums])
        self.course_feature_matrix = feature_matrix

        for idx, cid in enumerate(df["CourseID"]):
            self.course_id_to_idx[cid] = idx
            self.idx_to_course_id[idx] = cid

    def _build_cluster_affinity_matrix(self):
        """
        Computes the relative popularity and performance of each course within each cluster.
        """
        merged_tx = self.txns_df.merge(
            self.clustered_users_df[["UserID", "cluster_label"]], on="UserID", how="inner"
        )

        all_clusters = self.clustered_users_df["cluster_label"].dropna().unique()
        self.cluster_course_affinity = {}

        for c_id in all_clusters:
            c_tx = merged_tx[merged_tx["cluster_label"] == c_id]
            course_counts = c_tx["CourseID"].value_counts()
            max_count = course_counts.max() if not course_counts.empty else 1

            affinity_map = {}
            for cid in self.courses_df["CourseID"]:
                count = course_counts.get(cid, 0)
                # Normalized frequency within cluster
                affinity_map[cid] = float(count / max_count)

            self.cluster_course_affinity[int(c_id)] = affinity_map

    def get_user_enrolled_courses(self, user_id: str) -> List[str]:
        """Returns the list of course IDs a user is already enrolled in."""
        user_tx = self.txns_df[self.txns_df["UserID"] == user_id]
        return user_tx["CourseID"].tolist()

    def get_user_profile_vector(self, user_id: str) -> np.ndarray:
        """
        Builds user preference profile vector based on past enrolled courses.
        """
        enrolled_cids = self.get_user_enrolled_courses(user_id)
        if not enrolled_cids:
            # Cold-start / fallback to mean catalog vector
            return np.mean(self.course_feature_matrix, axis=0).reshape(1, -1)

        indices = [self.course_id_to_idx[cid] for cid in enrolled_cids if cid in self.course_id_to_idx]
        if not indices:
            return np.mean(self.course_feature_matrix, axis=0).reshape(1, -1)

        user_vec = np.mean(self.course_feature_matrix[indices], axis=0).reshape(1, -1)
        return user_vec

    def recommend(
        self,
        user_id: Optional[str] = None,
        cluster_id: Optional[int] = None,
        top_n: int = 5,
        exclude_enrolled: bool = True,
        category_filter: Optional[str] = None,
        level_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        min_rating: float = 0.0
    ) -> pd.DataFrame:
        """
        Generates personalized, rank-ordered course recommendations with explanations.
        """
        enrolled_cids = []
        if user_id:
            enrolled_cids = self.get_user_enrolled_courses(user_id)
            if cluster_id is None:
                user_row = self.clustered_users_df[self.clustered_users_df["UserID"] == user_id]
                if not user_row.empty:
                    cluster_id = int(user_row["cluster_label"].iloc[0])
                else:
                    cluster_id = 0
            user_vec = self.get_user_profile_vector(user_id)
        else:
            if cluster_id is None:
                cluster_id = 0
            user_vec = np.mean(self.course_feature_matrix, axis=0).reshape(1, -1)

        # 1. Content-based similarity scores
        cb_similarities = cosine_similarity(user_vec, self.course_feature_matrix)[0]

        # 2. Cluster affinity scores
        affinity_dict = self.cluster_course_affinity.get(cluster_id, {})

        # 3. Compute Composite Hybrid Scores
        scores = []
        max_rating = 5.0

        for idx, row in self.courses_df.iterrows():
            cid = row["CourseID"]
            
            # Filter checks
            if exclude_enrolled and cid in enrolled_cids:
                continue
            if category_filter and category_filter != "All" and row["CourseCategory"] != category_filter:
                continue
            if level_filter and level_filter != "All" and row["CourseLevel"] != level_filter:
                continue
            if type_filter and type_filter != "All" and row["CourseType"] != type_filter:
                continue
            if row["CourseRating"] < min_rating:
                continue

            cb_score = float(cb_similarities[idx])
            cl_score = float(affinity_dict.get(cid, 0.0))
            rt_score = float(row["CourseRating"] / max_rating)

            composite_score = (self.w_cb * cb_score) + (self.w_cl * cl_score) + (self.w_rt * rt_score)

            # Formulate intuitive recommendation rationale
            reasons = []
            if cl_score > 0.6:
                reasons.append(f"Top enrolled in your cluster segment")
            if cb_score > 0.7:
                reasons.append(f"High similarity to your learning history ({row['CourseCategory']})")
            if row["CourseRating"] >= 4.5:
                reasons.append(f"Top-rated course ({row['CourseRating']}★)")
            if row["CourseType"] == "Free":
                reasons.append("Free enrollment available")
            
            if not reasons:
                reasons.append(f"Recommended {row['CourseLevel']} {row['CourseCategory']} course")

            scores.append({
                "CourseID": cid,
                "CourseName": row["CourseName"],
                "CourseCategory": row["CourseCategory"],
                "CourseLevel": row["CourseLevel"],
                "CourseType": row["CourseType"],
                "CoursePrice": row["CoursePrice"],
                "CourseDuration": row["CourseDuration"],
                "CourseRating": row["CourseRating"],
                "MatchScore": round(float(composite_score) * 100, 1),
                "ContentSim": round(cb_score, 3),
                "ClusterAffinity": round(cl_score, 3),
                "RecommendationReason": " • ".join(reasons[:2])
            })

        if not scores:
            return pd.DataFrame()

        results_df = pd.DataFrame(scores)
        results_df = results_df.sort_values(by="MatchScore", ascending=False).head(top_n)
        return results_df


if __name__ == "__main__":
    recommender = HybridCourseRecommender()
    recs = recommender.recommend(user_id="U00003", top_n=5)
    print("Top Recommendations for User U00003:")
    print(recs[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "MatchScore", "RecommendationReason"]])

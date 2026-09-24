import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "Data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Raw data file paths
COURSES_FILE = DATA_DIR / "EduPro Online Platform.xlsx - Courses.csv"
USERS_FILE = DATA_DIR / "EduPro Online Platform.xlsx - Users.csv"
TRANSACTIONS_FILE = DATA_DIR / "EduPro Online Platform.xlsx - Transactions.csv"
TEACHERS_FILE = DATA_DIR / "EduPro Online Platform.xlsx - Teachers.csv"

# Processed data paths
PROCESSED_USER_FEATURES = ARTIFACTS_DIR / "learner_features.parquet"
CLUSTERED_USERS_FILE = ARTIFACTS_DIR / "clustered_learners.csv"
CLUSTER_SUMMARY_FILE = ARTIFACTS_DIR / "cluster_profiles.json"
RECOMMENDER_CACHE_FILE = ARTIFACTS_DIR / "course_similarity_matrix.npy"

# Model hyperparameter defaults
RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
MAX_CLUSTERS_EVAL = 8

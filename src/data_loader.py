import logging
from typing import Dict, Tuple
import pandas as pd
from .config import COURSES_FILE, USERS_FILE, TRANSACTIONS_FILE, TEACHERS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV files for Users, Courses, Transactions, and Teachers
    with appropriate datatype parsing and basic hygiene checks.
    """
    logger.info("Loading EduPro datasets from disk...")

    # 1. Load Courses
    courses_df = pd.read_csv(COURSES_FILE)
    courses_df["CourseID"] = courses_df["CourseID"].astype(str).str.strip()
    courses_df["CourseName"] = courses_df["CourseName"].astype(str).str.strip()
    courses_df["CourseCategory"] = courses_df["CourseCategory"].astype(str).str.strip()
    courses_df["CourseType"] = courses_df["CourseType"].astype(str).str.strip()
    courses_df["CourseLevel"] = courses_df["CourseLevel"].astype(str).str.strip()
    courses_df["CoursePrice"] = pd.to_numeric(courses_df["CoursePrice"], errors="coerce").fillna(0.0)
    courses_df["CourseDuration"] = pd.to_numeric(courses_df["CourseDuration"], errors="coerce").fillna(0.0)
    courses_df["CourseRating"] = pd.to_numeric(courses_df["CourseRating"], errors="coerce").fillna(0.0)

    # 2. Load Users
    users_df = pd.read_csv(USERS_FILE)
    users_df["UserID"] = users_df["UserID"].astype(str).str.strip()
    users_df["UserName"] = users_df["UserName"].astype(str).str.strip()
    users_df["Age"] = pd.to_numeric(users_df["Age"], errors="coerce")
    users_df["Gender"] = users_df["Gender"].astype(str).str.strip()
    users_df["Email"] = users_df["Email"].astype(str).str.strip()

    # 3. Load Teachers
    teachers_df = pd.read_csv(TEACHERS_FILE)
    teachers_df["TeacherID"] = teachers_df["TeacherID"].astype(str).str.strip()
    teachers_df["TeacherName"] = teachers_df["TeacherName"].astype(str).str.strip()
    teachers_df["Age"] = pd.to_numeric(teachers_df["Age"], errors="coerce")
    teachers_df["Gender"] = teachers_df["Gender"].astype(str).str.strip()
    teachers_df["Expertise"] = teachers_df["Expertise"].astype(str).str.strip()
    teachers_df["YearsOfExperience"] = pd.to_numeric(teachers_df["YearsOfExperience"], errors="coerce").fillna(0)
    teachers_df["TeacherRating"] = pd.to_numeric(teachers_df["TeacherRating"], errors="coerce").fillna(0.0)

    # 4. Load Transactions
    transactions_df = pd.read_csv(TRANSACTIONS_FILE)
    transactions_df["TransactionID"] = transactions_df["TransactionID"].astype(str).str.strip()
    transactions_df["UserID"] = transactions_df["UserID"].astype(str).str.strip()
    transactions_df["CourseID"] = transactions_df["CourseID"].astype(str).str.strip()
    transactions_df["TeacherID"] = transactions_df["TeacherID"].astype(str).str.strip()
    transactions_df["PaymentMethod"] = transactions_df["PaymentMethod"].astype(str).str.strip()
    transactions_df["Amount"] = pd.to_numeric(transactions_df["Amount"], errors="coerce").fillna(0.0)
    
    # Parse transaction dates with dayfirst=True
    transactions_df["TransactionDate"] = pd.to_datetime(
        transactions_df["TransactionDate"], format="%d/%m/%Y", errors="coerce"
    )

    logger.info(
        f"Datasets loaded successfully: Users ({len(users_df)}), "
        f"Courses ({len(courses_df)}), Transactions ({len(transactions_df)}), "
        f"Teachers ({len(teachers_df)})"
    )

    return courses_df, users_df, transactions_df, teachers_df


def get_merged_transaction_details() -> pd.DataFrame:
    """
    Enriches transactions with course, user, and teacher metadata for modeling and EDA.
    """
    courses_df, users_df, transactions_df, teachers_df = load_raw_data()

    merged = transactions_df.merge(users_df, on="UserID", how="left", suffixes=("", "_user"))
    merged = merged.merge(courses_df, on="CourseID", how="left", suffixes=("", "_course"))
    merged = merged.merge(teachers_df, on="TeacherID", how="left", suffixes=("", "_teacher"))

    return merged


if __name__ == "__main__":
    courses, users, txns, teachers = load_raw_data()
    print(f"Courses: {courses.shape}")
    print(f"Users: {users.shape}")
    print(f"Transactions: {txns.shape}")
    print(f"Teachers: {teachers.shape}")

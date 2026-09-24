# EduPro: Learner Intelligence & Personalization Engine

An end-to-end machine learning platform for unsupervised learner segmentation and cluster-aware course recommendations for EduPro Online Platform.

---

## 📌 Project Overview
Online education platforms host diverse learner demographics with varying motivations, proficiency levels, and budgets. Standard global course recommendations fail to optimize learner engagement and retention.

This project delivers:
1. **Multidimensional Feature Engineering:** Aggregates transactional telemetry into engagement, preference, and behavioral vectors.
2. **Unsupervised Learner Segmentation:** K-Means and Hierarchical Clustering identifying 4 actionable learner personas (*Foundational & Budget Learners*, *Advanced Domain Specialists*, *Intermediate Skill Builders*, *Career Up-skillers*).
3. **Cluster-Aware Hybrid Recommendation Engine:** Fuses content-based vector similarity, cluster collaborative affinity, and quality weighting.
4. **Streamlit Interactive Dashboard:** Real-time persona visualization, learner profile explorer, and adaptive course recommendations.

---

## 📂 Repository Structure

```
EDU_PRO_ONLINE/
├── Data/                              # Source datasets (Users, Courses, Transactions, Teachers)
│   ├── EduPro Online Platform.xlsx - Courses.csv
│   ├── EduPro Online Platform.xlsx - Teachers.csv
│   ├── EduPro Online Platform.xlsx - Transactions.csv
│   └── EduPro Online Platform.xlsx - Users.csv
├── src/                               # Modular source code
│   ├── __init__.py
│   ├── config.py                      # Paths and model constants
│   ├── data_loader.py                 # Ingestion & schema validation
│   ├── feature_engineering.py         # Learner feature aggregation
│   ├── clustering.py                  # K-Means, Hierarchical & PCA pipeline
│   ├── recommender.py                 # Hybrid recommendation engine
│   └── evaluation.py                  # System validation metrics
├── app/                               # Streamlit web application
│   └── main.py
├── artifacts/                         # Processed feature tables and metadata
├── models/                            # Serialized ML models and scalers
├── requirements.txt                   # Project dependencies
├── run_pipeline.py                    # Master pipeline runner
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Create and activate a virtual environment, then install dependencies:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline
Execute the full data ingestion, feature engineering, clustering, and evaluation pipeline:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
Launch the interactive Streamlit application:
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Evaluation & Benchmarks

| Metric | Baseline (Popularity) | EduPro Hybrid Engine | Lift / Status |
| :--- | :--- | :--- | :--- |
| **Optimal Partition (k)** | — | **k = 4** | Peak Silhouette ($0.382$) |
| **Davies-Bouldin Index** | — | **1.184** | Cohesive clusters |
| **Catalog Coverage** | 23.3% | **91.7%** | **+293.5%** discovery |
| **Category Precision** | 41.2% | **82.4%** | **+100.0%** relevance |
| **Mean Recommended Rating**| 3.10 ★ | **4.26 ★** | **+37.4%** quality boost |
| **Projected Engagement Lift**| Baseline | **+28.4%** | Retention boost |

---

## 👥 Authors & Acknowledgments
- **Tanay Dashore** —Unified Mentor Project.

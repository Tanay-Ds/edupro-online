import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Ensure src is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLUSTERED_USERS_FILE,
    CLUSTER_SUMMARY_FILE,
    MODELS_DIR,
    COURSES_FILE,
    USERS_FILE,
    TRANSACTIONS_FILE,
    TEACHERS_FILE
)
from src.data_loader import load_raw_data
from src.clustering import train_clustering_pipeline, evaluate_cluster_range
from src.feature_engineering import engineer_learner_features, get_clustering_feature_matrix
from src.recommender import HybridCourseRecommender
from src.evaluation import evaluate_segmentation_quality, evaluate_recommendation_performance

# Page configuration
st.set_page_config(
    page_title="EduPro | Learner Intelligence & Personalization Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .main-title-container {
        padding: 1.25rem 1.75rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -4px rgba(49, 46, 129, 0.25);
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    
    .persona-card {
        background: #ffffff;
        border-left: 5px solid #4f46e5;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .course-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .course-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.35rem;
    }
    .badge-primary { background: #e0e7ff; color: #3730a3; }
    .badge-secondary { background: #f1f5f9; color: #475569; }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-danger { background: #fee2e2; color: #991b1b; }
    
    .match-pill {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_data():
    courses_df, users_df, txns_df, teachers_df = load_raw_data()
    
    if not CLUSTERED_USERS_FILE.exists() or not CLUSTER_SUMMARY_FILE.exists():
        clustered_df, personas = train_clustering_pipeline()
    else:
        clustered_df = pd.read_csv(CLUSTERED_USERS_FILE)
        with open(CLUSTER_SUMMARY_FILE, "r") as f:
            personas = json.load(f)
            
    return courses_df, users_df, txns_df, teachers_df, clustered_df, personas


@st.cache_resource(show_spinner=False)
def get_recommender():
    return HybridCourseRecommender()


# Load data
courses_df, users_df, txns_df, teachers_df, clustered_df, personas = get_cached_data()
recommender = get_recommender()

# Header Banner
st.markdown("""
<div class="main-title-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.9rem; font-weight: 800; color: #ffffff;">EduPro Learner Intelligence Platform</h1>
            <p style="margin: 0.25rem 0 0 0; color: #c7d2fe; font-size: 0.95rem;">
                Personalized Course Recommendations & Unsupervised Learner Segmentation
            </p>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.12); padding: 0.5rem 1rem; border-radius: 8px;">
            <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #e0e7ff;">AI Engine</span>
            <div style="font-weight: 700; color: #34d399; font-size: 0.95rem;">● Active (k=4 Clusters)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation (3 Clean Tabs)
tab_choice = st.sidebar.radio(
    "Go To Workspace:",
    [
        "🎯 Personalized Recommender & Profiles",
        "🧩 Learner Segmentation & Personas",
        "📊 Platform & Content Analytics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 System Summary")
st.sidebar.markdown(f"""
- **Total Learners:** `{len(users_df):,}`
- **Active Transactions:** `{len(txns_df):,}`
- **Course Catalog:** `{len(courses_df)} courses`
- **Learner Segments:** `4 Personas`
""")


# ==============================================================================
# TAB 1: PERSONALIZED COURSE RECOMMENDER & LEARNER PROFILE (HERO WORKSPACE)
# ==============================================================================
if tab_choice == "🎯 Personalized Recommender & Profiles":
    st.subheader("🎯 Personalized Course Recommender & Learner Profile")
    st.markdown(
        "Select any learner to inspect their profile, view their assigned behavioral segment, and see tailored course recommendations."
    )

    # 1. Learner Selection Ribbon
    col_sel1, col_sel2 = st.columns([4, 8])
    with col_sel1:
        all_users = sorted(clustered_df["UserID"].unique())
        selected_uid = st.selectbox("👤 Select Learner ID:", all_users, index=0)
    
    with col_sel2:
        u_row = clustered_df[clustered_df["UserID"] == selected_uid].iloc[0]
        u_cluster = int(u_row["cluster_label"])
        u_persona = personas.get(str(u_cluster), {})
        
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1.2rem; margin-top: 0.35rem;">
            <b>Name:</b> {u_row['UserName']} &nbsp;|&nbsp; 
            <b>Demographics:</b> {u_row['Age']} yrs, {u_row['Gender']} &nbsp;|&nbsp; 
            <b>Assigned Segment:</b> <span style="color: #4338ca; font-weight: 700;">{u_persona.get('icon', '🎓')} {u_persona.get('title', 'Learner')}</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Learner Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Enrolled Courses", f"{int(u_row['total_courses_enrolled'])}")
    m2.metric("Total Spend", f"${u_row['total_spending']:.2f}")
    m3.metric("Paid Course Ratio", f"{u_row['paid_course_ratio']*100:.1f}%")
    m4.metric("Preferred Category", f"{u_row['preferred_category']}")
    m5.metric("Learning Depth", f"{u_row['learning_depth_index']:.2f}")

    st.markdown("---")

    # 3. Interactive Filter Ribbon for Recommendations
    st.markdown("#### 🔍 Filter Recommendations")
    f1, f2, f3 = st.columns(3)
    with f1:
        all_cats = ["All"] + sorted(courses_df["CourseCategory"].dropna().unique().tolist())
        filter_cat = st.selectbox("Category Filter:", all_cats)
    with f2:
        filter_lvl = st.selectbox("Difficulty Level:", ["All", "Beginner", "Intermediate", "Advanced"])
    with f3:
        filter_type = st.selectbox("Pricing Filter:", ["All", "Free", "Paid"])

    # Generate Recommendations
    recs = recommender.recommend(
        user_id=selected_uid,
        top_n=6,
        exclude_enrolled=True,
        category_filter=filter_cat,
        level_filter=filter_lvl,
        type_filter=filter_type
    )

    # 4. Display Recommendations Cards
    if recs.empty:
        st.warning("No recommendations match the specified filters. Try selecting 'All' for category or level.")
    else:
        st.markdown(f"#### 🌟 Top Recommended Courses for `{u_row['UserName']}`")
        cols = st.columns(2)
        for idx, (_, r) in enumerate(recs.iterrows()):
            with cols[idx % 2]:
                price_tag = f"${r['CoursePrice']:.2f}" if r['CourseType'] == "Paid" else "FREE"
                price_badge = "badge-danger" if r['CourseType'] == "Paid" else "badge-success"
                
                st.markdown(f"""
                <div class="course-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="badge badge-primary">{r['CourseCategory']}</span>
                            <span class="badge badge-secondary">{r['CourseLevel']}</span>
                            <span class="badge {price_badge}">{price_tag}</span>
                        </div>
                        <div class="match-pill">{r['MatchScore']}% Match</div>
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 0.5rem 0 0.2rem 0;">
                        {r['CourseName']}
                    </div>
                    <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem;">
                        Course ID: <code>{r['CourseID']}</code> • ⏱ {r['CourseDuration']} hrs • ⭐ <b>{r['CourseRating']}</b> / 5.0
                    </div>
                    <div style="background: #f8fafc; border-radius: 6px; padding: 0.45rem 0.65rem; font-size: 0.8rem; color: #334155; border-left: 3px solid #10b981;">
                        💡 <b>Why Recommended:</b> {r['RecommendationReason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 5. Enrolled History Collapsible
    with st.expander("📚 View Past Enrolled Courses for this Learner"):
        enrolled_ids = recommender.get_user_enrolled_courses(selected_uid)
        if enrolled_ids:
            enrolled_details = courses_df[courses_df["CourseID"].isin(enrolled_ids)]
            st.dataframe(
                enrolled_details[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseType", "CoursePrice", "CourseRating"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No prior enrollments found.")


# ==============================================================================
# TAB 2: LEARNER SEGMENTATION & PERSONAS (ML WORKSPACE)
# ==============================================================================
elif tab_choice == "🧩 Learner Segmentation & Personas":
    st.subheader("🧩 Unsupervised Learner Segmentation & Personas")
    st.markdown(
        "K-Means clustering grouped the platform's 3,000 learners into 4 distinct archetypes based on spending, depth, and activity."
    )

    # 4 Persona Cards
    p_cols = st.columns(4)
    for idx, (cid, p_data) in enumerate(personas.items()):
        with p_cols[idx]:
            m = p_data["metrics"]
            st.markdown(f"""
            <div class="persona-card">
                <div style="font-size: 1.6rem; margin-bottom: 0.2rem;">{p_data['icon']}</div>
                <div style="font-weight: 800; font-size: 0.95rem; color: #1e1b4b; min-height: 42px;">{p_data['title']}</div>
                <div style="font-size: 0.75rem; color: #6b7280; margin: 0.2rem 0;">Cluster #{cid} • <b>{p_data['share_pct']}%</b> ({p_data['size']:,} users)</div>
                <hr style="margin: 0.4rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <div style="font-size: 0.78rem; color: #374151; line-height: 1.35;">
                    • <b>Avg Spend:</b> ${m['avg_total_spending']:.2f}<br>
                    • <b>Avg Courses:</b> {m['avg_courses_enrolled']}<br>
                    • <b>Paid Ratio:</b> {m['paid_course_ratio']*100:.1f}%<br>
                    • <b>Domain:</b> {m['dominant_category']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2D/3D Cluster Visualization & Strategy Table
    c_map, c_radar = st.columns([6, 6])
    with c_map:
        st.markdown("#### 🌐 2D PCA Cluster Map")
        fig_pca = px.scatter(
            clustered_df,
            x="pca_x",
            y="pca_y",
            color="persona_title",
            hover_data=["UserID", "Age", "total_spending", "total_courses_enrolled", "preferred_category"],
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pca, use_container_width=True)

    with c_radar:
        st.markdown("#### 🎯 Cluster Centroid Feature Comparison")
        radar_metrics = [
            "avg_spending_per_course", "total_courses_enrolled", "paid_course_ratio",
            "diversity_score", "learning_depth_index", "avg_course_rating_enrolled"
        ]
        radar_df = clustered_df.groupby("persona_title")[radar_metrics].mean().reset_index()
        scaler = MinMaxScaler()
        scaled_vals = scaler.fit_transform(radar_df[radar_metrics])

        fig_radar = go.Figure()
        categories = ["Spend/Course", "Enrollments", "Paid Ratio", "Diversity", "Depth Index", "Avg Rating"]
        colors = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b"]

        for idx, row in radar_df.iterrows():
            vals = scaled_vals[idx].tolist()
            vals.append(vals[0])
            cats = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats,
                fill='toself',
                name=row["persona_title"][:18],
                line=dict(color=colors[idx % len(colors)])
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Strategy Table
    st.markdown("#### 📋 Recommended Actions Per Persona")
    strategy_rows = []
    for cid, p in personas.items():
        strategy_rows.append({
            "Cluster": f"Cluster {cid}",
            "Persona Archetype": p["title"],
            "Market Share": f"{p['share_pct']}% ({p['size']:,})",
            "Target Strategy": p["strategy"]
        })
    st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: PLATFORM & CONTENT ANALYTICS (BUSINESS & VALIDATION WORKSPACE)
# ==============================================================================
elif tab_choice == "📊 Platform & Content Analytics":
    st.subheader("📊 Platform Analytics & Model Validation")

    # 1. KPI Top Row
    k1, k2, k3, k4 = st.columns(4)
    total_rev = txns_df["Amount"].sum()
    total_enrollments = len(txns_df)
    avg_cat_rating = courses_df["CourseRating"].mean()

    X, feat_cols = get_clustering_feature_matrix(clustered_df)
    seg_metrics = evaluate_segmentation_quality(clustered_df, feat_cols)
    rec_metrics = evaluate_recommendation_performance(recommender, sample_size=100)

    k1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Revenue</div>
        <div class="metric-value">${total_rev:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    k2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Enrollments</div>
        <div class="metric-value">{total_enrollments:,}</div>
    </div>
    """, unsafe_allow_html=True)

    k3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Catalog Coverage</div>
        <div class="metric-value">{rec_metrics['catalog_coverage_pct']}%</div>
    </div>
    """, unsafe_allow_html=True)

    k4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Projected Lift</div>
        <div class="metric-value">+{rec_metrics['projected_engagement_lift_pct']}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### 📚 Catalog by Subject Domain")
        cat_dist = courses_df["CourseCategory"].value_counts().reset_index()
        cat_dist.columns = ["Category", "Count"]
        fig_cat = px.pie(cat_dist, values="Count", names="Category", hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism)
        fig_cat.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig_cat, use_container_width=True)

    with g2:
        st.markdown("#### 🏅 Top Rated Instructors")
        top_teachers = teachers_df.sort_values(by="TeacherRating", ascending=False).head(8)
        fig_tea = px.bar(
            top_teachers,
            x="TeacherRating",
            y="TeacherName",
            orientation="h",
            color="Expertise",
            text="TeacherRating",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_tea.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_tea, use_container_width=True)

    # 3. Model Validation Summary Table
    st.markdown("#### 🔬 Mathematical Validation Summary")
    val_table = pd.DataFrame([
        {"Validation Dimension": "Clustering Silhouette Score", "Value": f"{seg_metrics['silhouette_score']}", "Status": "Optimal partition (k=4)"},
        {"Validation Dimension": "Davies-Bouldin Index", "Value": f"{seg_metrics['davies_bouldin_index']}", "Status": "Well-separated clusters"},
        {"Validation Dimension": "Mean Recommended Course Rating", "Value": f"{rec_metrics['mean_recommended_course_rating']} ★", "Status": "High quality assurance"},
        {"Validation Dimension": "Catalog Coverage", "Value": f"{rec_metrics['catalog_coverage_pct']}%", "Status": "Long-tail course discovery"},
        {"Validation Dimension": "Estimated Engagement Lift", "Value": f"+{rec_metrics['projected_engagement_lift_pct']}%", "Status": "Platform retention boost"}
    ])
    st.dataframe(val_table, use_container_width=True, hide_index=True)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]
SPEND_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
CLUSTER_NAMES = {
    0: "High-Value Active Multichannel Customers",
    1: "Low-Value Inactive / At-Risk Customers",
    2: "High-Income / VIP Customers",
    3: "Low-Value Recently Active Customers",
}
SEGMENT_ORDER = [CLUSTER_NAMES[i] for i in sorted(CLUSTER_NAMES)]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, ["#2a9d8f", "#e76f51", "#e9c46a", "#457b9d"]))

PREDICTOR = "predictor"
DASHBOARD = "dashboard"

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

kmeans = joblib.load(BASE_DIR / "kmeans_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

if "page" not in st.session_state:
    st.session_state.page = PREDICTOR


def go_to(page):
    st.session_state.page = page


# ---------------------------------------------------------------
# Data for the dashboard
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the dataset, clean it like the notebook, and assign segments
    with the saved scaler and K-Means model."""
    df = pd.read_csv(BASE_DIR / "customer_segmentation.csv")
    df = df.dropna()
    df["Age"] = 2026 - df["Year_Birth"]
    df = df[df["Age"] < 100].copy()
    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)

    X_scaled = scaler.transform(df[FEATURES])
    df["Cluster"] = kmeans.predict(X_scaled)
    df["Customer_Segment"] = df["Cluster"].map(CLUSTER_NAMES)

    pca_data = PCA(n_components=2).fit_transform(X_scaled)
    df["PCA1"] = pca_data[:, 0]
    df["PCA2"] = pca_data[:, 1]

    bins = [18, 30, 40, 50, 60, 70, 100]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
    return df


def show_fig(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------
# Page 1 - Predictor
# ---------------------------------------------------------------
def predictor_page():
    top_left, top_right = st.columns([4, 1])
    top_left.title("Customer Segmentation App")
    top_right.button(
        "📊 Open Dashboard",
        on_click=go_to,
        args=(DASHBOARD,),
        width="stretch",
        key="to_dashboard",
    )

    st.write("Enter customer details to predict their segment.")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (in $)", min_value=0, max_value=200000, value=50000)
    total_spent = st.number_input("Total Spending (in $)", min_value=0, max_value=5000, value=1000)
    num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
    num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
    num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=20)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Income": [income],
            "Total_Spending": [total_spent],
            "NumWebPurchases": [num_web_purchases],
            "NumStorePurchases": [num_store_purchases],
            "NumWebVisitsMonth": [num_web_visits],
            "Recency": [recency],
        }
    )

    input_data_scaled = scaler.transform(input_data)

    if st.button("Predict Customer Segment"):
        cluster = kmeans.predict(input_data_scaled)[0]
        segment_name = CLUSTER_NAMES[cluster]
        st.success(f"The predicted customer segment is: **{segment_name}**")


# ---------------------------------------------------------------
# Page 2 - Dashboard
# ---------------------------------------------------------------
def dashboard_page():
    top_left, top_right = st.columns([4, 1])
    top_left.title("Customer Segmentation Dashboard")
    top_right.button(
        "🔮 Back to Predictor",
        on_click=go_to,
        args=(PREDICTOR,),
        width="stretch",
        key="to_predictor",
    )

    data = load_data()

    selected = st.multiselect(
        "Filter by segment",
        options=SEGMENT_ORDER,
        default=SEGMENT_ORDER,
    )
    df = data[data["Customer_Segment"].isin(selected)]
    if df.empty:
        st.warning("Select at least one segment to see the dashboard.")
        return

    # ---- KPIs ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Avg. income", f"${df['Income'].mean():,.0f}")
    k3.metric("Avg. total spending", f"${df['Total_Spending'].mean():,.0f}")
    k4.metric("Avg. recency", f"{df['Recency'].mean():.0f} days")
    k5.metric("Avg. age", f"{df['Age'].mean():.0f}")

    st.divider()

    # ---- Segmentation ----
    st.subheader("Segmentation")
    c1, c2 = st.columns(2)

    with c1:
        counts = df["Customer_Segment"].value_counts().reindex(SEGMENT_ORDER).dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(counts.index, counts.values, color=[SEGMENT_COLORS[s] for s in counts.index])
        ax.invert_yaxis()
        total = counts.sum()
        for y, v in enumerate(counts.values):
            ax.text(v, y, f"  {v:,} ({v / total:.0%})", va="center", fontsize=9)
        ax.set_xlim(0, counts.max() * 1.3)
        ax.set_title("Customers per segment")
        ax.set_xlabel("Customers")
        show_fig(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x="PCA1",
            y="PCA2",
            hue="Customer_Segment",
            hue_order=[s for s in SEGMENT_ORDER if s in selected],
            palette=SEGMENT_COLORS,
            s=25,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title("Customer segments (PCA projection)")
        ax.legend(title=None, fontsize=7, loc="best")
        show_fig(fig)

    st.markdown("**Segment profile** (average of each feature)")
    profile = (
        df.groupby("Customer_Segment")[FEATURES]
        .mean()
        .reindex([s for s in SEGMENT_ORDER if s in selected])
        .round(1)
    )
    profile.insert(0, "Customers", df["Customer_Segment"].value_counts())
    st.dataframe(profile, width="stretch")

    st.divider()

    # ---- Analysis ----
    st.subheader("Analysis")
    a1, a2 = st.columns(2)

    with a1:
        spend = df.groupby("Customer_Segment")["Total_Spending"].mean().reindex(
            [s for s in SEGMENT_ORDER if s in selected]
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(spend.index, spend.values, color=[SEGMENT_COLORS[s] for s in spend.index])
        ax.invert_yaxis()
        ax.set_title("Average total spending by segment")
        ax.set_xlabel("Total spending ($)")
        show_fig(fig)

    with a2:
        channels = (
            df.groupby("Customer_Segment")[["NumWebPurchases", "NumStorePurchases"]]
            .mean()
            .reindex([s for s in SEGMENT_ORDER if s in selected])
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        y = np.arange(len(channels))
        ax.barh(y - 0.2, channels["NumWebPurchases"], height=0.4, label="Web", color="#264653")
        ax.barh(y + 0.2, channels["NumStorePurchases"], height=0.4, label="Store", color="#f4a261")
        ax.set_yticks(y)
        ax.set_yticklabels(channels.index)
        ax.invert_yaxis()
        ax.set_title("Average purchases by channel")
        ax.set_xlabel("Purchases per customer")
        ax.legend()
        show_fig(fig)

    a3, a4 = st.columns(2)

    with a3:
        income_age = df.groupby("AgeGroup", observed=True)["Income"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(income_age.index.astype(str), income_age.values, color="#2a9d8f")
        ax.set_title("Average income by age group")
        ax.set_xlabel("Age group")
        ax.set_ylabel("Income ($)")
        show_fig(fig)

    with a4:
        edu = df.groupby("Education")["Total_Spending"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(edu.index, edu.values, color="#457b9d")
        ax.set_title("Average total spending by education")
        ax.set_xlabel("Total spending ($)")
        show_fig(fig)

    a5, a6 = st.columns(2)

    with a5:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Total_Spending"], bins=30, kde=True, color="#e76f51", ax=ax)
        ax.set_title("Total spending distribution")
        ax.set_xlabel("Total spending ($)")
        show_fig(fig)

    with a6:
        corr = df[
            ["Income", "Age", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "Recency"]
        ].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlation between key features")
        show_fig(fig)


# ---------------------------------------------------------------
# Router
# ---------------------------------------------------------------
if st.session_state.page == DASHBOARD:
    dashboard_page()
else:
    predictor_page()
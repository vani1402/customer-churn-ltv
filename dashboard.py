import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Customer Churn & LTV Dashboard",
    page_icon="📊",
    layout="wide"
)

# Find the LTV dataset
project_folder = Path(__file__).parent
files = list(project_folder.rglob("customer_ltv_dataset.csv"))

if not files:
    st.error("customer_ltv_dataset.csv was not found.")
    st.stop()

# Load data
df = pd.read_csv(files[0])

# Title
st.title("📊 Customer Churn & Lifetime Value Dashboard")
st.write("Customer Churn Prediction & LTV Analysis")

# Sidebar
st.sidebar.header("Filters")

# Contract filter
contracts = st.sidebar.multiselect(
    "Select Contract",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

# LTV filter
ltv_categories = st.sidebar.multiselect(
    "Select LTV Category",
    options=df["LTV_Category"].unique(),
    default=df["LTV_Category"].unique()
)

# Churn filter
churn_options = st.sidebar.multiselect(
    "Select Churn Status",
    options=df["Churn"].unique(),
    default=df["Churn"].unique()
)

# Apply filters
filtered_df = df[
    (df["Contract"].isin(contracts)) &
    (df["LTV_Category"].isin(ltv_categories)) &
    (df["Churn"].isin(churn_options))
]

# KPI calculations
total_customers = len(filtered_df)

churned_customers = (
    filtered_df["Churn"] == "Yes"
).sum()

if total_customers > 0:
    churn_rate = churned_customers / total_customers * 100
else:
    churn_rate = 0

average_ltv = filtered_df["LTV"].mean() if total_customers > 0 else 0

high_ltv = (
    filtered_df["LTV_Category"] == "High"
).sum()

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Average LTV",
    f"{average_ltv:,.2f}"
)

st.divider()

# -----------------------------
# PIE CHART
# -----------------------------

st.subheader("1. Customer Churn Distribution")

churn_counts = filtered_df["Churn"].value_counts()

fig1, ax1 = plt.subplots()

ax1.pie(
    churn_counts.values,
    labels=churn_counts.index,
    autopct="%1.1f%%"
)

ax1.set_title("Churn vs No Churn")

st.pyplot(fig1)

# -----------------------------
# CONTRACT BAR CHART
# -----------------------------

st.subheader("2. Churn by Contract Type")

contract_churn = pd.crosstab(
    filtered_df["Contract"],
    filtered_df["Churn"],
    normalize="index"
) * 100

fig2, ax2 = plt.subplots()

contract_churn.plot(
    kind="bar",
    ax=ax2
)

ax2.set_ylabel("Percentage")
ax2.set_xlabel("Contract")
ax2.set_title("Churn Percentage by Contract")

plt.xticks(rotation=0)

st.pyplot(fig2)

# -----------------------------
# TENURE BAR CHART
# -----------------------------

st.subheader("3. Churn by Tenure Group")

tenure_churn = pd.crosstab(
    filtered_df["TenureGroup"],
    filtered_df["Churn"],
    normalize="index"
) * 100

fig3, ax3 = plt.subplots()

tenure_churn.plot(
    kind="bar",
    ax=ax3
)

ax3.set_ylabel("Percentage")
ax3.set_xlabel("Tenure Group")
ax3.set_title("Churn Percentage by Tenure")

plt.xticks(rotation=0)

st.pyplot(fig3)

# -----------------------------
# LTV BAR CHART
# -----------------------------

st.subheader("4. Customer Distribution by LTV Category")

ltv_counts = filtered_df["LTV_Category"].value_counts()

fig4, ax4 = plt.subplots()

ltv_counts.plot(
    kind="bar",
    ax=ax4
)

ax4.set_ylabel("Number of Customers")
ax4.set_xlabel("LTV Category")
ax4.set_title("LTV Category Distribution")

plt.xticks(rotation=0)

st.pyplot(fig4)

# -----------------------------
# HIGH VALUE CHURN
# -----------------------------

st.subheader("5. High-LTV Customers Who Churned")

high_value_churn = filtered_df[
    (filtered_df["LTV_Category"] == "High") &
    (filtered_df["Churn"] == "Yes")
]

st.metric(
    "High-LTV Customers Lost",
    f"{len(high_value_churn):,}"
)

# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------

st.subheader("💡 Business Insights")

st.write(
    f"• The dashboard currently contains {total_customers:,} customers."
)

st.write(
    f"• The current churn rate is {churn_rate:.2f}%."
)

st.write(
    f"• Average customer LTV is {average_ltv:,.2f}."
)

st.write(
    f"• {len(high_value_churn):,} high-LTV customers have churned."
)
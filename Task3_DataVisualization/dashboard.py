import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM DASHBOARD STYLE
# ============================================================

st.markdown("""
<style>
/* Main application background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f7fa;
}

/* Centered page title */
h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.dashboard-subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 20px;
}

/* KPI cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e5ea;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.08);
}

/* Fix KPI text visibility in dark/light themes */
[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #eef3f7;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    width: 100%;
}

/* Insight boxes */
.insight-box {
    background-color: white;
    padding: 16px 20px;
    border-radius: 10px;
    border-left: 5px solid #2e7d32;
    margin-bottom: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv("dataset/sales.csv")
except FileNotFoundError:
    st.error("❌ sales.csv was not found inside the dataset folder.")
    st.info("Make sure your project structure is: dataset/sales.csv")
    st.stop()

required_columns = [
    "Order Date",
    "Category",
    "Sub-Category",
    "Region",
    "Sales",
    "Quantity",
    "Profit"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"❌ Missing columns in sales.csv: {missing_columns}")
    st.stop()

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Remove rows with invalid dates
df = df.dropna(subset=["Order Date"]).copy()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔍 Dashboard Filters")
st.sidebar.write("Use the filters below to explore the dataset.")

categories = sorted(df["Category"].dropna().unique())

selected_category = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

regions = sorted(df["Region"].dropna().unique())

selected_region = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    (df["Category"].isin(selected_category)) &
    (df["Region"].isin(selected_region))
].copy()

if len(selected_dates) == 2:
    start_date = pd.to_datetime(selected_dates[0])
    end_date = pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1)

    filtered_df = filtered_df[
        (filtered_df["Order Date"] >= start_date) &
        (filtered_df["Order Date"] < end_date)
    ].copy()

if filtered_df.empty:
    st.warning(
        "⚠️ No data is available for the selected filters. "
        "Please select different filters."
    )
    st.stop()

# ============================================================
# TITLE
# ============================================================

st.title("📊 Sales Analysis Dashboard")

st.markdown(
    "<div class='dashboard-subtitle'>"
    "Interactive Sales, Profit & Performance Analysis"
    "</div>",
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_quantity = filtered_df["Quantity"].sum()
average_sales = filtered_df["Sales"].mean()
profit_margin = (total_profit / total_sales * 100) if total_sales else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Sales",
    f"₹{total_sales:,.0f}"
)

col2.metric(
    "📈 Total Profit",
    f"₹{total_profit:,.0f}"
)

col3.metric(
    "📦 Quantity Sold",
    f"{total_quantity:,}"
)

col4.metric(
    "💵 Average Sale",
    f"₹{average_sales:,.2f}"
)

st.caption(f"Profit Margin: {profit_margin:.2f}%")

st.divider()

# ============================================================
# KEY INSIGHTS & RECOMMENDATIONS
# ============================================================

st.header("💡 Key Insights & Recommendations")

category_sales = filtered_df.groupby("Category")["Sales"].sum()
region_sales = filtered_df.groupby("Region")["Sales"].sum()
subcategory_sales = filtered_df.groupby("Sub-Category")["Sales"].sum()

best_category = category_sales.idxmax()
best_category_sales = category_sales.max()

best_region = region_sales.idxmax()
best_region_sales = region_sales.max()

best_subcategory = subcategory_sales.idxmax()
best_subcategory_sales = subcategory_sales.max()

lowest_category = category_sales.idxmin()
lowest_region = region_sales.idxmin()

st.subheader("📌 Key Findings")

st.markdown(
    f"<div class='insight-box'>"
    f"📊 <b>{best_category}</b> is the highest-performing category "
    f"with sales of <b>₹{best_category_sales:,.0f}</b>."
    f"</div>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div class='insight-box'>"
    f"🌍 <b>{best_region}</b> is the highest-performing region "
    f"with sales of <b>₹{best_region_sales:,.0f}</b>."
    f"</div>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div class='insight-box'>"
    f"🏆 <b>{best_subcategory}</b> is the leading sub-category "
    f"with sales of <b>₹{best_subcategory_sales:,.0f}</b>."
    f"</div>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div class='insight-box'>"
    f"💰 The selected data contains total sales of "
    f"<b>₹{total_sales:,.0f}</b> and total profit of "
    f"<b>₹{total_profit:,.0f}</b>."
    f"</div>",
    unsafe_allow_html=True
)

st.subheader("🎯 Recommendations")

st.write(
    f"• Focus marketing and promotional activities on "
    f"**{best_category}**, the highest-selling category."
)

st.write(
    f"• Continue strengthening strategies in **{best_region}** "
    f"while investigating opportunities in **{lowest_region}**."
)

st.write(
    f"• Give special attention to **{best_subcategory}**, "
    f"the leading sub-category."
)

st.write(
    "• Use the monthly sales trend to identify high-performing "
    "periods and plan future campaigns."
)

st.divider()

# ============================================================
# SALES BY CATEGORY
# ============================================================

st.header("📊 Sales by Category")

fig1, ax1 = plt.subplots(figsize=(9, 5))

bars = ax1.bar(
    category_sales.index,
    category_sales.values
)

ax1.set_xlabel("Category")
ax1.set_ylabel("Sales (₹)")
ax1.set_title("Sales by Category")

for bar, value in zip(bars, category_sales.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"₹{value:,.0f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ============================================================
# MONTHLY SALES
# ============================================================

st.header("📈 Monthly Sales")

monthly_sales = (
    filtered_df.assign(
        Month=filtered_df["Order Date"].dt.to_period("M").astype(str)
    )
    .groupby("Month")["Sales"]
    .sum()
)

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

ax2.set_xlabel("Month")
ax2.set_ylabel("Sales (₹)")
ax2.set_title("Monthly Sales Trend")

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ============================================================
# SALES BY REGION
# ============================================================

st.header("🥧 Sales by Region")

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.pie(
    region_sales.values,
    labels=region_sales.index,
    autopct="%1.1f%%",
    startangle=90
)

ax3.set_title("Sales by Region")

plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ============================================================
# PROFIT DISTRIBUTION
# ============================================================

st.header("📊 Profit Distribution")

fig4, ax4 = plt.subplots(figsize=(8, 5))

ax4.hist(
    filtered_df["Profit"],
    bins=10
)

ax4.set_xlabel("Profit (₹)")
ax4.set_ylabel("Number of Orders")
ax4.set_title("Profit Distribution")

plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# ============================================================
# SALES VS PROFIT
# ============================================================

st.header("🔵 Sales vs Profit")

fig5, ax5 = plt.subplots(figsize=(8, 5))

ax5.scatter(
    filtered_df["Sales"],
    filtered_df["Profit"]
)

ax5.set_xlabel("Sales (₹)")
ax5.set_ylabel("Profit (₹)")
ax5.set_title("Sales vs Profit Relationship")

plt.tight_layout()
st.pyplot(fig5)
plt.close(fig5)

# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.header("🔥 Correlation Heatmap")

correlation = filtered_df[
    ["Sales", "Quantity", "Profit"]
].corr()

fig6, ax6 = plt.subplots(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax6
)

ax6.set_title("Sales, Quantity and Profit Correlation")

plt.tight_layout()
st.pyplot(fig6)
plt.close(fig6)

# ============================================================
# DATASET SUMMARY
# ============================================================

st.header("📋 Dataset Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric(
    "Rows",
    f"{len(filtered_df):,}"
)

summary_col2.metric(
    "Columns",
    f"{filtered_df.shape[1]:,}"
)

summary_col3.metric(
    "Duplicate Rows",
    f"{filtered_df.duplicated().sum():,}"
)

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True
)

# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

st.header("📥 Download Data")

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="filtered_sales.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    "<div class='footer'>"
    "Sales Analysis Dashboard | Data Visualization Task 3"
    "</div>",
    unsafe_allow_html=True
)

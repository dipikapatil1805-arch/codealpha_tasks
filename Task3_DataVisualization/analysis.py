from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset/sales.csv")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create charts folder if it does not exist
os.makedirs("charts", exist_ok=True)


# ==============================
# DATASET INFORMATION
# ==============================

print("===== DATASET INFORMATION =====")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==============================
# BASIC SALES ANALYSIS
# ==============================

print("\n===== SALES ANALYSIS =====")

# Total Sales
total_sales = df["Sales"].sum()
print("\nTotal Sales:", total_sales)

# Total Profit
total_profit = df["Profit"].sum()
print("Total Profit:", total_profit)

# Total Quantity
total_quantity = df["Quantity"].sum()
print("Total Quantity Sold:", total_quantity)

# Average Sales
average_sales = df["Sales"].mean()
print("Average Sale:", round(average_sales, 2))


# ==============================
# CATEGORY ANALYSIS
# ==============================

print("\n===== SALES BY CATEGORY =====")

category_sales = df.groupby("Category")["Sales"].sum()

print(category_sales)


# ==============================
# REGION ANALYSIS
# ==============================

print("\n===== SALES BY REGION =====")

region_sales = df.groupby("Region")["Sales"].sum()

print(region_sales)


# ==============================
# SUB-CATEGORY ANALYSIS
# ==============================

print("\n===== SALES BY SUB-CATEGORY =====")

subcategory_sales = df.groupby("Sub-Category")["Sales"].sum()

print(subcategory_sales)


# ==============================
# HYPOTHESIS TESTING
# ==============================

print("\n===== HYPOTHESIS TESTING =====")

# Question:
# Is there a significant difference in sales
# between Technology and non-Technology products?

technology_sales = df[df["Category"] == "Technology"]["Sales"]

other_sales = df[df["Category"] != "Technology"]["Sales"]

print("\nResearch Question:")
print(
    "Is there a significant difference in sales between "
    "Technology and other categories?"
)

# Null Hypothesis (H0)
print("\nNull Hypothesis (H0):")
print(
    "There is no significant difference in average sales "
    "between Technology and other categories."
)

# Alternative Hypothesis (H1)
print("\nAlternative Hypothesis (H1):")
print(
    "There is a significant difference in average sales "
    "between Technology and other categories."
)

# Perform independent t-test
t_stat, p_value = ttest_ind(
    technology_sales,
    other_sales,
    equal_var=False
)

print("\nTechnology Average Sales:",
      round(technology_sales.mean(), 2))

print("Other Categories Average Sales:",
      round(other_sales.mean(), 2))

print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

# Significance level
alpha = 0.05

print("\nSignificance Level:", alpha)

if p_value < alpha:
    print("\nResult:")
    print("Reject the Null Hypothesis.")
    print(
        "There is statistically significant evidence of "
        "a difference in average sales."
    )
else:
    print("\nResult:")
    print("Fail to Reject the Null Hypothesis.")
    print(
        "There is not enough statistical evidence of "
        "a difference in average sales."
    )


# ==============================
# BAR CHART - SALES BY CATEGORY
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    category_sales.index,
    category_sales.values
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("charts/sales_by_category.png")

plt.show()


# ==============================
# LINE CHART - MONTHLY SALES
# ==============================

df["Month"] = (
    df["Order Date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("charts/monthly_sales.png")

plt.show()


# ==============================
# PIE CHART - SALES BY REGION
# ==============================

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(7, 7))

plt.pie(
    region_sales.values,
    labels=region_sales.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sales by Region")

plt.tight_layout()

plt.savefig("charts/sales_by_region.png")

plt.show()


# ==============================
# HISTOGRAM - PROFIT DISTRIBUTION
# ==============================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Profit"],
    bins=10
)

plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Number of Orders")

plt.tight_layout()

plt.savefig("charts/profit_distribution.png")

plt.show()


# ==============================
# SCATTER PLOT - SALES VS PROFIT
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Sales"],
    df["Profit"]
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("charts/sales_vs_profit.png")

plt.show()


# ==============================
# HEATMAP - CORRELATION
# ==============================

plt.figure(figsize=(8, 6))

correlation = df[
    ["Sales", "Quantity", "Profit"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Sales, Quantity and Profit Correlation")

plt.tight_layout()

plt.savefig("charts/correlation_heatmap.png")

plt.show()


# ==============================
# FINAL MESSAGE
# ==============================

print("\n===== ANALYSIS COMPLETED SUCCESSFULLY =====")
print("All analysis and visualizations have been generated.")
print("Charts are saved in the 'charts' folder.")
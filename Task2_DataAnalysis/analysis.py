"""
=============================================================================
TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Explores the dataset structure, identifies trends/patterns/anomalies,
tests hypotheses using statistics, and detects data issues.
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_PATH = "dataset/books.csv"
OUTPUT_DIR = "outputs"

# Rating text-to-number mapping
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def separator(title):
    """Print a formatted section separator."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def load_and_clean_data():
    """Load the dataset and perform data cleaning."""
    separator("LOADING & CLEANING DATA")

    df = pd.read_csv(DATASET_PATH)
    print(f"\n📂 Loaded dataset: {DATASET_PATH}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Data Structure ---
    print(f"\n📊 Column Information:")
    print(f"   {'Column':<20} {'Dtype':<15} {'Non-Null':<10} {'Unique':<8}")
    print(f"   {'─' * 53}")
    for col in df.columns:
        print(f"   {col:<20} {str(df[col].dtype):<15} {df[col].notna().sum():<10} {df[col].nunique():<8}")

    # --- Missing Values ---
    print(f"\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values found!")
    else:
        for col, count in missing[missing > 0].items():
            print(f"   ⚠ {col}: {count} missing ({count/len(df)*100:.1f}%)")

    # --- Duplicate Check ---
    dupes = df.duplicated().sum()
    print(f"\n🔁 Duplicate Rows: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"   → Removed {dupes} duplicates. New shape: {df.shape}")

    # --- Convert Rating to Numeric ---
    if df["Rating"].dtype == "object":
        df["Rating_Numeric"] = df["Rating"].map(RATING_MAP)
        print(f"\n🔄 Converted 'Rating' text to numeric (1-5)")

    # --- Ensure Price is numeric ---
    if df["Price"].dtype == "object":
        df["Price"] = df["Price"].replace(r"[£Â]", "", regex=True).astype(float)
        print("🔄 Cleaned 'Price' column to numeric")

    print(f"\n✅ Data cleaning complete! Final shape: {df.shape}")
    return df


def explore_data_structure(df):
    """Explore variables and data types in depth."""
    separator("DATA STRUCTURE EXPLORATION")

    # Data types summary
    print(f"\n📋 Data Types Summary:")
    print(df.dtypes.to_string())

    # First and last rows
    print(f"\n📖 First 5 Records:")
    print(df.head().to_string())

    print(f"\n📖 Last 5 Records:")
    print(df.tail().to_string())

    # Unique values per column
    print(f"\n🔢 Unique Values:")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"   {col}: {unique_vals} unique values")
        if unique_vals <= 10:
            print(f"      → {df[col].unique().tolist()}")


def statistical_analysis(df):
    """Perform descriptive and inferential statistical analysis."""
    separator("STATISTICAL ANALYSIS")

    # --- Descriptive Statistics ---
    print("\n📊 Descriptive Statistics for Price:")
    price = df["Price"]
    print(f"   Count  : {price.count()}")
    print(f"   Mean   : £{price.mean():.2f}")
    print(f"   Median : £{price.median():.2f}")
    print(f"   Mode   : £{price.mode().values[0]:.2f}")
    print(f"   Std Dev: £{price.std():.2f}")
    print(f"   Min    : £{price.min():.2f}")
    print(f"   Max    : £{price.max():.2f}")
    print(f"   Range  : £{price.max() - price.min():.2f}")
    print(f"   Skew   : {price.skew():.3f}")
    print(f"   Kurtosis: {price.kurtosis():.3f}")

    # Quartiles
    print(f"\n📐 Price Quartiles:")
    for q in [0.25, 0.50, 0.75]:
        print(f"   Q{int(q*100):>2}%  : £{price.quantile(q):.2f}")

    # --- Rating Distribution ---
    print(f"\n⭐ Rating Distribution:")
    rating_col = "Rating_Numeric" if "Rating_Numeric" in df.columns else "Rating"
    rating_counts = df[rating_col].value_counts().sort_index()
    for rating, count in rating_counts.items():
        bar = "█" * count
        pct = count / len(df) * 100
        print(f"   Rating {rating}: {count:>3} ({pct:>5.1f}%) {bar}")

    # --- Price by Rating ---
    print(f"\n💰 Average Price by Rating:")
    price_by_rating = df.groupby(rating_col)["Price"].agg(["mean", "median", "count"])
    for rating, row in price_by_rating.iterrows():
        print(f"   Rating {rating}: Mean=£{row['mean']:.2f}, Median=£{row['median']:.2f}, Count={int(row['count'])}")

    # Full describe
    print(f"\n📈 Full Statistical Summary:")
    print(df.describe().to_string())

    return price_by_rating


def identify_trends_and_patterns(df):
    """Identify trends, patterns, and correlations in the data."""
    separator("TRENDS, PATTERNS & CORRELATIONS")

    rating_col = "Rating_Numeric" if "Rating_Numeric" in df.columns else "Rating"

    # --- Correlation ---
    if df[rating_col].dtype in ["int64", "float64"]:
        corr = df["Price"].corr(df[rating_col])
        print(f"\n📈 Correlation between Price and Rating: {corr:.4f}")
        if abs(corr) < 0.1:
            print("   → Very weak correlation — price and rating are mostly independent")
        elif abs(corr) < 0.3:
            print("   → Weak correlation")
        elif abs(corr) < 0.5:
            print("   → Moderate correlation")
        else:
            print("   → Strong correlation")

    # --- Price Distribution Patterns ---
    price = df["Price"]
    print(f"\n💰 Price Distribution Patterns:")
    price_ranges = [
        ("Budget (< £20)", price[price < 20]),
        ("Mid-range (£20-£40)", price[(price >= 20) & (price < 40)]),
        ("Premium (£40-£55)", price[(price >= 40) & (price < 55)]),
        ("Expensive (≥ £55)", price[price >= 55]),
    ]
    for label, subset in price_ranges:
        pct = len(subset) / len(df) * 100
        print(f"   {label:<25}: {len(subset):>3} books ({pct:>5.1f}%)")

    # --- Rating Patterns ---
    print(f"\n⭐ Rating Patterns:")
    most_common = df[rating_col].mode().values[0]
    least_common = df[rating_col].value_counts().idxmin()
    print(f"   Most common rating  : {most_common}")
    print(f"   Least common rating : {least_common}")

    # --- Availability ---
    print(f"\n📦 Availability Breakdown:")
    avail_counts = df["Availability"].value_counts()
    for avail, count in avail_counts.items():
        print(f"   {avail}: {count} ({count/len(df)*100:.1f}%)")

    # --- Title Length Analysis ---
    df["Title_Length"] = df["Title"].str.len()
    print(f"\n📝 Title Length Analysis:")
    print(f"   Shortest title: {df['Title_Length'].min()} chars — \"{df.loc[df['Title_Length'].idxmin(), 'Title']}\"")
    print(f"   Longest title : {df['Title_Length'].max()} chars — \"{df.loc[df['Title_Length'].idxmax(), 'Title'][:60]}...\"")
    print(f"   Average length: {df['Title_Length'].mean():.1f} chars")

    return df


def detect_anomalies(df):
    """Detect outliers and anomalies using IQR method."""
    separator("ANOMALY & OUTLIER DETECTION")

    price = df["Price"]

    # IQR Method
    Q1 = price.quantile(0.25)
    Q3 = price.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\n📊 IQR Outlier Detection for Price:")
    print(f"   Q1 (25th percentile) : £{Q1:.2f}")
    print(f"   Q3 (75th percentile) : £{Q3:.2f}")
    print(f"   IQR                  : £{IQR:.2f}")
    print(f"   Lower Bound          : £{lower_bound:.2f}")
    print(f"   Upper Bound          : £{upper_bound:.2f}")

    outliers = df[(price < lower_bound) | (price > upper_bound)]
    print(f"\n   Outliers found: {len(outliers)}")
    if len(outliers) > 0:
        print(f"\n   Outlier Books:")
        for _, row in outliers.iterrows():
            print(f"   ⚠ £{row['Price']:.2f} — {row['Title'][:50]}")
    else:
        print("   ✓ No price outliers detected using IQR method")

    # Z-Score method
    z_scores = np.abs(stats.zscore(price))
    z_outliers = df[z_scores > 2]
    print(f"\n📊 Z-Score Outlier Detection (|z| > 2):")
    print(f"   Outliers found: {len(z_outliers)}")
    if len(z_outliers) > 0:
        for _, row in z_outliers.iterrows():
            z = abs((row["Price"] - price.mean()) / price.std())
            print(f"   ⚠ £{row['Price']:.2f} (z={z:.2f}) — {row['Title'][:50]}")

    return outliers


def hypothesis_testing(df):
    """Test hypotheses using statistical methods."""
    separator("HYPOTHESIS TESTING")

    rating_col = "Rating_Numeric" if "Rating_Numeric" in df.columns else "Rating"

    # Hypothesis 1: Do higher-rated books (4-5) have different prices than lower-rated (1-2)?
    print("\n📋 Hypothesis 1: Higher-rated books have different prices than lower-rated books")
    print("   H₀: No significant difference in mean price between high and low rated books")
    print("   H₁: Significant difference exists\n")

    high_rated = df[df[rating_col] >= 4]["Price"]
    low_rated = df[df[rating_col] <= 2]["Price"]

    if len(high_rated) >= 2 and len(low_rated) >= 2:
        t_stat, p_value = stats.ttest_ind(high_rated, low_rated, equal_var=False)
        alpha = 0.05

        print(f"   High-rated (4-5) mean price: £{high_rated.mean():.2f} (n={len(high_rated)})")
        print(f"   Low-rated  (1-2) mean price: £{low_rated.mean():.2f} (n={len(low_rated)})")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value:     {p_value:.4f}")
        print(f"   Significance level (α): {alpha}")

        if p_value < alpha:
            print(f"\n   ✓ RESULT: Reject H₀ — There IS a significant difference (p={p_value:.4f} < {alpha})")
        else:
            print(f"\n   ✗ RESULT: Fail to reject H₀ — No significant difference (p={p_value:.4f} ≥ {alpha})")
    else:
        print("   ⚠ Insufficient data for t-test")

    # Hypothesis 2: Is price normally distributed?
    print(f"\n{'─' * 70}")
    print("\n📋 Hypothesis 2: Book prices follow a normal distribution")
    print("   H₀: Prices are normally distributed")
    print("   H₁: Prices are NOT normally distributed\n")

    shapiro_stat, shapiro_p = stats.shapiro(df["Price"])
    print(f"   Shapiro-Wilk test statistic: {shapiro_stat:.4f}")
    print(f"   p-value:                     {shapiro_p:.4f}")

    if shapiro_p < 0.05:
        print(f"\n   ✓ RESULT: Reject H₀ — Prices are NOT normally distributed")
    else:
        print(f"\n   ✗ RESULT: Fail to reject H₀ — Prices appear normally distributed")

    # Hypothesis 3: ANOVA — Is there a significant difference in price across all ratings?
    print(f"\n{'─' * 70}")
    print("\n📋 Hypothesis 3: Price differs significantly across rating groups (ANOVA)")
    print("   H₀: Mean prices are equal across all rating groups")
    print("   H₁: At least one group has a different mean price\n")

    groups = [group["Price"].values for name, group in df.groupby(rating_col) if len(group) >= 2]
    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   p-value:     {anova_p:.4f}")

        if anova_p < 0.05:
            print(f"\n   ✓ RESULT: Reject H₀ — Prices differ significantly across ratings")
        else:
            print(f"\n   ✗ RESULT: Fail to reject H₀ — No significant difference across ratings")


def detect_data_issues(df):
    """Detect potential data quality issues."""
    separator("DATA ISSUES & PROBLEMS")

    issues_found = 0

    # Check for negative prices
    neg_prices = df[df["Price"] < 0]
    if len(neg_prices) > 0:
        print(f"   ⚠ {len(neg_prices)} negative prices found!")
        issues_found += 1
    else:
        print("   ✓ No negative prices")

    # Check for zero prices
    zero_prices = df[df["Price"] == 0]
    if len(zero_prices) > 0:
        print(f"   ⚠ {len(zero_prices)} zero-priced books found!")
        issues_found += 1
    else:
        print("   ✓ No zero prices")

    # Check for empty titles
    empty_titles = df[df["Title"].str.strip() == ""]
    if len(empty_titles) > 0:
        print(f"   ⚠ {len(empty_titles)} empty titles found!")
        issues_found += 1
    else:
        print("   ✓ No empty titles")

    # Check for duplicate titles
    dup_titles = df[df["Title"].duplicated(keep=False)]
    if len(dup_titles) > 0:
        print(f"   ⚠ {len(dup_titles)} duplicate titles found!")
        issues_found += 1
    else:
        print("   ✓ No duplicate titles")

    # Check for invalid ratings
    valid_ratings = {"One", "Two", "Three", "Four", "Five"}
    if df["Rating"].dtype == "object":
        invalid = df[~df["Rating"].isin(valid_ratings)]
        if len(invalid) > 0:
            print(f"   ⚠ {len(invalid)} invalid ratings found!")
            issues_found += 1
        else:
            print("   ✓ All ratings are valid")

    # Summary
    if issues_found == 0:
        print(f"\n   🎉 No data quality issues detected!")
    else:
        print(f"\n   ⚠ {issues_found} issue(s) need attention")


def save_eda_summary(df):
    """Save EDA results to a summary file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")

    rating_col = "Rating_Numeric" if "Rating_Numeric" in df.columns else "Rating"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS (EDA) — SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Records: {len(df)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write(df.describe().to_string())
        f.write(f"\n\nRating Distribution:\n")
        f.write(df[rating_col].value_counts().sort_index().to_string())
        f.write(f"\n\nPrice by Rating:\n")
        f.write(df.groupby(rating_col)["Price"].describe().to_string())

    print(f"\n💾 EDA summary saved to: {summary_path}")


def main():
    """Run the complete EDA pipeline."""
    print("=" * 70)
    print("  TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("  Dataset: Books from books.toscrape.com")
    print("=" * 70)

    # Step 1: Load and clean data
    df = load_and_clean_data()

    # Step 2: Explore data structure
    explore_data_structure(df)

    # Step 3: Statistical analysis
    statistical_analysis(df)

    # Step 4: Identify trends and patterns
    df = identify_trends_and_patterns(df)

    # Step 5: Detect anomalies
    detect_anomalies(df)

    # Step 6: Hypothesis testing
    hypothesis_testing(df)

    # Step 7: Data issues
    detect_data_issues(df)

    # Step 8: Save summary
    save_eda_summary(df)

    print(f"\n{'═' * 70}")
    print("  ✅ EDA COMPLETE!")
    print(f"{'═' * 70}\n")

    return df


if __name__ == "__main__":
    main()
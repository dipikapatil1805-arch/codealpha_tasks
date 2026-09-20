"""
=============================================================================
TASK 3: DATA VISUALIZATION
=============================================================================
Transforms raw data into visual formats — charts, graphs, and dashboards.
Uses Matplotlib and Seaborn to create impactful visualizations.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_PATH = "dataset/books.csv"
OUTPUT_DIR = "outputs"

# Rating text-to-number mapping
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# Style settings
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


def load_data():
    """Load and prepare data for visualization."""
    df = pd.read_csv(DATASET_PATH)
    df["Rating_Numeric"] = df["Rating"].map(RATING_MAP)
    if df["Price"].dtype == "object":
        df["Price"] = df["Price"].replace(r"[£Â]", "", regex=True).astype(float)
    df["Title_Length"] = df["Title"].str.len()
    return df


def plot_price_distribution(df):
    """1. Price Distribution — Histogram with KDE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Price"], bins=20, kde=True, color=COLORS[0], edgecolor="white", ax=ax)
    ax.axvline(df["Price"].mean(), color="red", linestyle="--", linewidth=2, label=f'Mean: £{df["Price"].mean():.2f}')
    ax.axvline(df["Price"].median(), color="green", linestyle="--", linewidth=2, label=f'Median: £{df["Price"].median():.2f}')
    ax.set_title("Distribution of Book Prices", fontsize=16, fontweight="bold")
    ax.set_xlabel("Price (£)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)
    path = os.path.join(OUTPUT_DIR, "01_price_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_rating_distribution(df):
    """2. Rating Distribution — Bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rating_counts = df["Rating_Numeric"].value_counts().sort_index()
    bars = ax.bar(rating_counts.index, rating_counts.values, color=COLORS, edgecolor="white", width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Rating Distribution of Books", fontsize=16, fontweight="bold")
    ax.set_xlabel("Rating (Stars)", fontsize=12)
    ax.set_ylabel("Number of Books", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["★ 1", "★★ 2", "★★★ 3", "★★★★ 4", "★★★★★ 5"])
    path = os.path.join(OUTPUT_DIR, "02_rating_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_price_vs_rating(df):
    """3. Price vs Rating — Box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Rating_Numeric", y="Price", data=df, palette="Set2", ax=ax, width=0.5)
    sns.stripplot(x="Rating_Numeric", y="Price", data=df, color="black", alpha=0.4, size=4, ax=ax)

    ax.set_title("Price Distribution by Rating", fontsize=16, fontweight="bold")
    ax.set_xlabel("Rating (Stars)", fontsize=12)
    ax.set_ylabel("Price (£)", fontsize=12)
    ax.set_xticklabels(["★ 1", "★★ 2", "★★★ 3", "★★★★ 4", "★★★★★ 5"])
    path = os.path.join(OUTPUT_DIR, "03_price_vs_rating.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_top_expensive_books(df):
    """4. Top 10 Most Expensive Books — Horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    top10 = df.nlargest(10, "Price")

    # Shorten long titles
    titles = [t[:45] + "..." if len(t) > 45 else t for t in top10["Title"]]

    bars = ax.barh(range(len(titles)), top10["Price"].values, color=sns.color_palette("Reds_r", 10), edgecolor="white")

    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=10)
    ax.invert_yaxis()

    # Add price labels
    for bar, price in zip(bars, top10["Price"].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"£{price:.2f}", va="center", fontweight="bold", fontsize=10)

    ax.set_title("Top 10 Most Expensive Books", fontsize=16, fontweight="bold")
    ax.set_xlabel("Price (£)", fontsize=12)
    path = os.path.join(OUTPUT_DIR, "04_top10_expensive.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_top_cheapest_books(df):
    """5. Top 10 Cheapest Books — Horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom10 = df.nsmallest(10, "Price")

    titles = [t[:45] + "..." if len(t) > 45 else t for t in bottom10["Title"]]

    bars = ax.barh(range(len(titles)), bottom10["Price"].values, color=sns.color_palette("Greens_r", 10), edgecolor="white")

    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=10)
    ax.invert_yaxis()

    for bar, price in zip(bars, bottom10["Price"].values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"£{price:.2f}", va="center", fontweight="bold", fontsize=10)

    ax.set_title("Top 10 Cheapest Books", fontsize=16, fontweight="bold")
    ax.set_xlabel("Price (£)", fontsize=12)
    path = os.path.join(OUTPUT_DIR, "05_top10_cheapest.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_correlation_heatmap(df):
    """6. Correlation Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df[["Price", "Rating_Numeric", "Title_Length"]].copy()
    numeric_df.columns = ["Price (£)", "Rating", "Title Length"]
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                mask=mask, square=True, linewidths=2, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})

    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
    path = os.path.join(OUTPUT_DIR, "06_correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_availability_pie(df):
    """7. Availability Pie Chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    avail_counts = df["Availability"].value_counts()

    wedges, texts, autotexts = ax.pie(
        avail_counts.values,
        labels=avail_counts.index,
        autopct="%1.1f%%",
        colors=COLORS[:len(avail_counts)],
        startangle=90,
        explode=[0.05] * len(avail_counts),
        shadow=True,
        textprops={"fontsize": 12}
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title("Book Availability", fontsize=16, fontweight="bold")
    path = os.path.join(OUTPUT_DIR, "07_availability_pie.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_price_range_bar(df):
    """8. Price Range Distribution."""
    fig, ax = plt.subplots(figsize=(9, 6))

    bins = [0, 15, 25, 35, 45, 60]
    labels = ["£0-15", "£15-25", "£25-35", "£35-45", "£45-60"]
    df["Price_Range"] = pd.cut(df["Price"], bins=bins, labels=labels, include_lowest=True)

    range_counts = df["Price_Range"].value_counts().sort_index()
    bars = ax.bar(range_counts.index, range_counts.values, color=COLORS, edgecolor="white", width=0.6)

    for bar, val in zip(bars, range_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Books by Price Range", fontsize=16, fontweight="bold")
    ax.set_xlabel("Price Range", fontsize=12)
    ax.set_ylabel("Number of Books", fontsize=12)
    path = os.path.join(OUTPUT_DIR, "08_price_range.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_dashboard(df):
    """9. Combined Dashboard — Multi-panel overview."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("📊 Book Dataset — Dashboard Overview", fontsize=20, fontweight="bold", y=1.02)

    # Panel 1: Price Distribution
    sns.histplot(df["Price"], bins=15, kde=True, color=COLORS[0], ax=axes[0, 0], edgecolor="white")
    axes[0, 0].set_title("Price Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Price (£)")

    # Panel 2: Rating Distribution
    rating_counts = df["Rating_Numeric"].value_counts().sort_index()
    axes[0, 1].bar(rating_counts.index, rating_counts.values, color=COLORS, edgecolor="white")
    axes[0, 1].set_title("Rating Distribution", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Rating")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_xticks(range(1, 6))

    # Panel 3: Box plot Price by Rating
    sns.boxplot(x="Rating_Numeric", y="Price", data=df, palette="Set2", ax=axes[1, 0])
    axes[1, 0].set_title("Price by Rating", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Rating")
    axes[1, 0].set_ylabel("Price (£)")

    # Panel 4: Price scatter with rating color
    scatter = axes[1, 1].scatter(range(len(df)), df["Price"], c=df["Rating_Numeric"],
                                  cmap="viridis", alpha=0.7, edgecolors="white", s=60)
    axes[1, 1].set_title("Price per Book (colored by Rating)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Book Index")
    axes[1, 1].set_ylabel("Price (£)")
    plt.colorbar(scatter, ax=axes[1, 1], label="Rating")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("  TASK 3: DATA VISUALIZATION")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    print(f"\n📊 Generating visualizations from {len(df)} books...\n")

    plot_price_distribution(df)
    plot_rating_distribution(df)
    plot_price_vs_rating(df)
    plot_top_expensive_books(df)
    plot_top_cheapest_books(df)
    plot_correlation_heatmap(df)
    plot_availability_pie(df)
    plot_price_range_bar(df)
    plot_dashboard(df)

    print(f"\n{'─' * 70}")
    print(f"✅ ALL VISUALIZATIONS SAVED to '{OUTPUT_DIR}/' directory!")
    print(f"   Total charts generated: 9")
    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()


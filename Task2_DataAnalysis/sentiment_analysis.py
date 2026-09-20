"""
=============================================================================
TASK 4: SENTIMENT ANALYSIS
=============================================================================
Analyzes book titles using NLP to classify sentiment as positive, negative,
or neutral. Uses TextBlob for polarity/subjectivity scoring.
Explores relationships between title sentiment and price/rating.
=============================================================================
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_PATH = "dataset/books.csv"
OUTPUT_DIR = "outputs"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATASET_PATH)
    df["Rating_Numeric"] = df["Rating"].map(RATING_MAP)
    if df["Price"].dtype == "object":
        df["Price"] = df["Price"].replace(r"[£Â]", "", regex=True).astype(float)
    return df


def analyze_sentiment(text):
    """Analyze sentiment of a text using TextBlob NLP."""
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity        # -1 (negative) to +1 (positive)
    subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

    # Classify sentiment
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, subjectivity, sentiment


def run_sentiment_analysis(df):
    """Apply sentiment analysis to all book titles."""
    print("\n🔍 Analyzing sentiment of book titles using TextBlob NLP...\n")

    results = df["Title"].apply(lambda t: pd.Series(analyze_sentiment(t),
                                                      index=["Polarity", "Subjectivity", "Sentiment"]))
    df = pd.concat([df, results], axis=1)

    # Summary
    sentiment_counts = df["Sentiment"].value_counts()
    total = len(df)

    print(f"   📊 Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        bar = "█" * int(count / total * 40)
        pct = count / total * 100
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "")
        print(f"   {emoji} {sentiment:<10}: {count:>3} ({pct:>5.1f}%) {bar}")

    print(f"\n   📈 Polarity Statistics:")
    print(f"   Mean polarity     : {df['Polarity'].mean():.4f}")
    print(f"   Median polarity   : {df['Polarity'].median():.4f}")
    print(f"   Std deviation     : {df['Polarity'].std():.4f}")
    print(f"   Min polarity      : {df['Polarity'].min():.4f}")
    print(f"   Max polarity      : {df['Polarity'].max():.4f}")

    print(f"\n   📈 Subjectivity Statistics:")
    print(f"   Mean subjectivity : {df['Subjectivity'].mean():.4f}")
    print(f"   Median subjectivity: {df['Subjectivity'].median():.4f}")

    return df


def show_example_sentiments(df):
    """Display example titles for each sentiment category."""
    print(f"\n{'─' * 70}")
    print("📋 Example Titles by Sentiment:")

    for sentiment in ["Positive", "Negative", "Neutral"]:
        subset = df[df["Sentiment"] == sentiment]
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[sentiment]
        print(f"\n   {emoji} {sentiment} Titles (showing up to 5):")
        for _, row in subset.head(5).iterrows():
            title = row["Title"][:55] + "..." if len(row["Title"]) > 55 else row["Title"]
            print(f"      • {title}")
            print(f"        Polarity: {row['Polarity']:.3f} | Subjectivity: {row['Subjectivity']:.3f}")


def analyze_sentiment_vs_features(df):
    """Analyze relationship between sentiment and price/rating."""
    print(f"\n{'─' * 70}")
    print("📊 Sentiment vs. Book Features:\n")

    # Sentiment vs Price
    print("   💰 Average Price by Sentiment:")
    price_by_sentiment = df.groupby("Sentiment")["Price"].agg(["mean", "median", "count"])
    for sentiment, row in price_by_sentiment.iterrows():
        print(f"   {sentiment:<10}: Mean=£{row['mean']:.2f}, Median=£{row['median']:.2f}, Count={int(row['count'])}")

    # Sentiment vs Rating
    print(f"\n   ⭐ Average Rating by Sentiment:")
    rating_by_sentiment = df.groupby("Sentiment")["Rating_Numeric"].agg(["mean", "median"])
    for sentiment, row in rating_by_sentiment.iterrows():
        print(f"   {sentiment:<10}: Mean={row['mean']:.2f}, Median={row['median']:.1f}")

    # Rating vs Sentiment
    print(f"\n   📊 Sentiment Distribution by Rating:")
    cross = pd.crosstab(df["Rating_Numeric"], df["Sentiment"], normalize="index") * 100
    print(cross.round(1).to_string())


def plot_sentiment_distribution(df):
    """Visualize sentiment distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    sentiment_counts = df["Sentiment"].value_counts()
    colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    pie_colors = [colors.get(s, "#999") for s in sentiment_counts.index]

    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                autopct="%1.1f%%", colors=pie_colors, startangle=90,
                explode=[0.05] * len(sentiment_counts), shadow=True,
                textprops={"fontsize": 12, "fontweight": "bold"})
    axes[0].set_title("Sentiment Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[1].bar(sentiment_counts.index, sentiment_counts.values,
                       color=pie_colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, sentiment_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=13)
    axes[1].set_title("Sentiment Counts", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Books")

    plt.suptitle("Book Title Sentiment Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_sentiment_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n   ✓ Saved: {path}")


def plot_polarity_distribution(df):
    """Plot polarity score distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Polarity"], bins=25, kde=True, color="#2196F3", edgecolor="white", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Neutral (0)")
    ax.axvline(df["Polarity"].mean(), color="green", linestyle="--", linewidth=2,
               label=f'Mean: {df["Polarity"].mean():.3f}')
    ax.set_title("Distribution of Polarity Scores", fontsize=16, fontweight="bold")
    ax.set_xlabel("Polarity Score (-1 = Negative, +1 = Positive)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)
    path = os.path.join(OUTPUT_DIR, "11_polarity_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_sentiment_vs_rating(df):
    """Plot sentiment polarity vs book rating."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    for sentiment, color in colors_map.items():
        subset = df[df["Sentiment"] == sentiment]
        ax.scatter(subset["Rating_Numeric"], subset["Polarity"],
                   c=color, label=sentiment, alpha=0.7, s=80, edgecolors="white")

    ax.set_title("Sentiment Polarity vs. Book Rating", fontsize=16, fontweight="bold")
    ax.set_xlabel("Rating (Stars)", fontsize=12)
    ax.set_ylabel("Polarity Score", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    path = os.path.join(OUTPUT_DIR, "12_sentiment_vs_rating.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_sentiment_vs_price(df):
    """Plot sentiment polarity vs price."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    for sentiment, color in colors_map.items():
        subset = df[df["Sentiment"] == sentiment]
        ax.scatter(subset["Price"], subset["Polarity"],
                   c=color, label=sentiment, alpha=0.7, s=80, edgecolors="white")

    ax.set_title("Sentiment Polarity vs. Price", fontsize=16, fontweight="bold")
    ax.set_xlabel("Price (£)", fontsize=12)
    ax.set_ylabel("Polarity Score", fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    path = os.path.join(OUTPUT_DIR, "13_sentiment_vs_price.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def plot_sentiment_dashboard(df):
    """Combined sentiment analysis dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("📊 Sentiment Analysis Dashboard", fontsize=20, fontweight="bold", y=1.02)

    colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}

    # 1. Sentiment counts
    sentiment_counts = df["Sentiment"].value_counts()
    pie_colors = [colors_map.get(s, "#999") for s in sentiment_counts.index]
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct="%1.1f%%", colors=pie_colors, startangle=90)
    axes[0, 0].set_title("Sentiment Split", fontweight="bold")

    # 2. Polarity histogram
    sns.histplot(df["Polarity"], bins=20, kde=True, color="#2196F3", ax=axes[0, 1], edgecolor="white")
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set_title("Polarity Distribution", fontweight="bold")

    # 3. Sentiment by Rating (stacked bar)
    cross = pd.crosstab(df["Rating_Numeric"], df["Sentiment"])
    cross_order = [c for c in ["Positive", "Neutral", "Negative"] if c in cross.columns]
    cross[cross_order].plot(kind="bar", stacked=True, ax=axes[1, 0],
                             color=[colors_map[c] for c in cross_order], edgecolor="white")
    axes[1, 0].set_title("Sentiment by Rating", fontweight="bold")
    axes[1, 0].set_xlabel("Rating")
    axes[1, 0].legend(title="Sentiment")
    axes[1, 0].tick_params(axis="x", rotation=0)

    # 4. Polarity vs Price scatter
    for sentiment, color in colors_map.items():
        subset = df[df["Sentiment"] == sentiment]
        axes[1, 1].scatter(subset["Price"], subset["Polarity"],
                           c=color, label=sentiment, alpha=0.6, s=50, edgecolors="white")
    axes[1, 1].set_title("Polarity vs Price", fontweight="bold")
    axes[1, 1].set_xlabel("Price (£)")
    axes[1, 1].set_ylabel("Polarity")
    axes[1, 1].legend()
    axes[1, 1].axhline(0, color="gray", linestyle=":")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "14_sentiment_dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   ✓ Saved: {path}")


def save_results(df):
    """Save sentiment analysis results to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "sentiment_results.csv")
    output_cols = ["Title", "Price", "Rating", "Rating_Numeric",
                   "Polarity", "Subjectivity", "Sentiment"]
    df[output_cols].to_csv(output_path, index=False)
    print(f"\n   💾 Results saved to: {output_path}")


def main():
    """Run the complete sentiment analysis pipeline."""
    print("=" * 70)
    print("  TASK 4: SENTIMENT ANALYSIS")
    print("  Analyzing book titles using NLP (TextBlob)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data()
    print(f"\n📂 Loaded {len(df)} books from {DATASET_PATH}")

    # Run sentiment analysis
    df = run_sentiment_analysis(df)

    # Show examples
    show_example_sentiments(df)

    # Analyze relationships
    analyze_sentiment_vs_features(df)

    # Generate visualizations
    print(f"\n{'─' * 70}")
    print("📊 Generating Sentiment Visualizations...\n")
    plot_sentiment_distribution(df)
    plot_polarity_distribution(df)
    plot_sentiment_vs_rating(df)
    plot_sentiment_vs_price(df)
    plot_sentiment_dashboard(df)

    # Save results
    save_results(df)

    print(f"\n{'═' * 70}")
    print("  ✅ SENTIMENT ANALYSIS COMPLETE!")
    print(f"{'═' * 70}\n")

    return df


if __name__ == "__main__":
    main()


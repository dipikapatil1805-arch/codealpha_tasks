# 📊 Data Analysis Project — Books Dataset

A comprehensive data analysis project covering **Web Scraping**, **Exploratory Data Analysis (EDA)**, **Data Visualization**, and **Sentiment Analysis** using a books dataset from [books.toscrape.com](https://books.toscrape.com).

---

## 📁 Project Structure

```
DataAnalysisTask2/
├── dataset/
│   ├── books.csv              # Original dataset (101 books)
│   └── books_scraped.csv      # Scraped dataset with extra fields
├── outputs/
│   ├── eda_summary.txt        # EDA summary report
│   ├── sentiment_results.csv  # Sentiment analysis results
│   ├── 01_price_distribution.png
│   ├── 02_rating_distribution.png
│   ├── 03_price_vs_rating.png
│   ├── 04_top10_expensive.png
│   ├── 05_top10_cheapest.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_availability_pie.png
│   ├── 08_price_range.png
│   ├── 09_dashboard.png
│   ├── 10_sentiment_distribution.png
│   ├── 11_polarity_distribution.png
│   ├── 12_sentiment_vs_rating.png
│   ├── 13_sentiment_vs_price.png
│   └── 14_sentiment_dashboard.png
├── scraper.py                 # Task 1: Web Scraping
├── analysis.py                # Task 2: Exploratory Data Analysis
├── visualizations.py          # Task 3: Data Visualization
├── sentiment_analysis.py      # Task 4: Sentiment Analysis
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation

1. **Install Python 3.8+** if not already installed.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for Sentiment Analysis):
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('brown')"
   ```

---

## 🚀 How to Run

### Task 1: Web Scraping
```bash
python scraper.py
```
- Scrapes books from **books.toscrape.com** using **BeautifulSoup** and **Requests**
- Extracts: Title, Price, Rating, Availability, Category, Description, Book URL
- Handles pagination and error handling
- Saves output to `dataset/books_scraped.csv`

### Task 2: Exploratory Data Analysis (EDA)
```bash
python analysis.py
```
- Loads and cleans the dataset
- Explores data structure (dtypes, shape, unique values)
- Performs descriptive statistics (mean, median, mode, std, quartiles)
- Identifies trends, patterns, and correlations
- Detects outliers using IQR and Z-Score methods
- Tests hypotheses (t-test, Shapiro-Wilk, ANOVA)
- Checks for data quality issues
- Saves summary report to `outputs/eda_summary.txt`

### Task 3: Data Visualization
```bash
python visualizations.py
```
Creates **9 visualizations** using **Matplotlib** and **Seaborn**:
1. Price Distribution (Histogram + KDE)
2. Rating Distribution (Bar Chart)
3. Price vs Rating (Box Plot)
4. Top 10 Expensive Books (Horizontal Bar)
5. Top 10 Cheapest Books (Horizontal Bar)
6. Correlation Heatmap
7. Availability Pie Chart
8. Price Range Distribution
9. Combined Dashboard

All saved as PNG files in `outputs/`.

### Task 4: Sentiment Analysis
```bash
python sentiment_analysis.py
```
- Uses **TextBlob** (NLP) to analyze book title sentiment
- Classifies each title as **Positive**, **Negative**, or **Neutral**
- Computes polarity and subjectivity scores
- Analyzes sentiment vs. price and rating relationships
- Creates **5 sentiment visualizations**
- Saves results to `outputs/sentiment_results.csv`

---

## 📦 Dependencies

| Package         | Purpose                        |
|-----------------|--------------------------------|
| pandas          | Data manipulation & analysis   |
| numpy           | Numerical operations           |
| matplotlib      | Data visualization             |
| seaborn         | Statistical visualizations     |
| requests        | HTTP requests for web scraping |
| beautifulsoup4  | HTML parsing                   |
| textblob        | NLP / Sentiment analysis       |
| scipy           | Statistical testing            |

---

## 📊 Dataset

- **Source**: [books.toscrape.com](https://books.toscrape.com)
- **Records**: 101 books
- **Columns**: Title, Price, Rating, Availability, Book URL
- **Extended dataset** (scraped): Also includes Category and Description

---

## ✅ Key Findings

- Price range: £10.16 — £58.11
- Ratings are distributed across all 5 levels
- Price and rating show weak correlation (mostly independent)
- Most book titles have neutral sentiment
- No significant data quality issues detected


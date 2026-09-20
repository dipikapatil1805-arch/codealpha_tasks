"""
=============================================================================
TASK 1: WEB SCRAPING — Books from books.toscrape.com
=============================================================================
Uses BeautifulSoup and Requests to scrape book data from a public website.
Handles HTML structure, pagination, and web navigation.
Creates a custom dataset tailored to analysis needs.
=============================================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
BOOK_BASE_URL = "https://books.toscrape.com/catalogue/"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "books_scraped.csv")

# Rating text-to-number mapping
RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}


def get_page(url, retries=2):
    """Fetch a web page with retry logic and error handling."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"  ⚠ Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            time.sleep(2)
    print(f"  ✗ Failed to fetch: {url}")
    return None


def scrape_book_details(book_url):
    """Scrape additional details from an individual book page."""
    response = get_page(book_url)
    if not response:
        return {"Category": "Unknown", "Description": "N/A"}

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract category from breadcrumb
    breadcrumb = soup.find("ul", class_="breadcrumb")
    category = "Unknown"
    if breadcrumb:
        links = breadcrumb.find_all("a")
        if len(links) >= 3:
            category = links[2].text.strip()

    # Extract book description
    description = "N/A"
    desc_tag = soup.find("div", id="product_description")
    if desc_tag:
        desc_p = desc_tag.find_next_sibling("p")
        if desc_p:
            description = desc_p.text.strip()

    return {"Category": category, "Description": description}


def scrape_books_page(page_num):
    """Scrape all books from a single catalogue page."""
    url = BASE_URL.format(page_num)
    response = get_page(url)
    if not response:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    books = []

    articles = soup.find_all("article", class_="product_pod")
    if not articles:
        return []

    for article in articles:
        # Title
        title_tag = article.find("h3").find("a")
        title = title_tag["title"]

        # Price
        price_text = article.find("p", class_="price_color").text
        price = float(price_text.replace("£", "").replace("Â", "").strip())

        # Rating
        rating_tag = article.find("p", class_="star-rating")
        rating_class = rating_tag["class"][1]  # e.g., "Three"
        rating = RATING_MAP.get(rating_class, 0)

        # Availability
        avail_tag = article.find("p", class_="instock")
        availability = avail_tag.text.strip() if avail_tag else "Unknown"

        # Book detail URL
        book_href = title_tag["href"]
        if not book_href.startswith("http"):
            book_url = BOOK_BASE_URL + book_href
        else:
            book_url = book_href

        books.append({
            "Title": title,
            "Price": price,
            "Rating": rating_class,
            "Rating_Numeric": rating,
            "Availability": availability,
            "Book_URL": book_url
        })

    return books


def offline_fallback():
    """Fallback: use existing books.csv and enrich it when scraping is not possible."""
    print("\n⚠ Network unreachable. Using OFFLINE FALLBACK mode...")
    print("  → Reading existing dataset and enriching it.\n")

    existing_csv = os.path.join(OUTPUT_DIR, "books.csv")
    if not os.path.exists(existing_csv):
        print(f"  ✗ No existing dataset found at {existing_csv}")
        return None

    df = pd.read_csv(existing_csv)

    # Add Rating_Numeric column
    df["Rating_Numeric"] = df["Rating"].map(RATING_MAP)

    # Add Category (derived from title keywords as approximation)
    def guess_category(title):
        title_lower = title.lower()
        if any(w in title_lower for w in ["cook", "recipe", "kitchen", "food", "diet"]):
            return "Food and Drink"
        elif any(w in title_lower for w in ["poem", "sonnets", "verse"]):
            return "Poetry"
        elif any(w in title_lower for w in ["history", "america", "world", "war", "politics"]):
            return "History"
        elif any(w in title_lower for w in ["art", "music", "drawing", "pencil", "paint"]):
            return "Art"
        elif any(w in title_lower for w in ["self", "life", "soul", "mind", "magic", "joy"]):
            return "Self Help"
        elif any(w in title_lower for w in ["novel", "fiction", "story", "mystery", "thriller"]):
            return "Fiction"
        elif any(w in title_lower for w in ["science", "technology", "digital", "online"]):
            return "Science"
        elif any(w in title_lower for w in ["comic", "manga", "volume", "vol.", "omnibus"]):
            return "Comics"
        else:
            return "General"

    df["Category"] = df["Title"].apply(guess_category)

    # Add placeholder description
    df["Description"] = "Description not available (offline mode)"

    # Rename URL column for consistency
    if "Book URL" in df.columns:
        df = df.rename(columns={"Book URL": "Book_URL"})

    # Reorder columns
    columns = ["Title", "Price", "Rating", "Rating_Numeric", "Availability",
               "Category", "Description", "Book_URL"]
    df = df[[c for c in columns if c in df.columns]]

    return df


def main():
    """Main scraping function — scrapes all 50 pages of the catalogue."""
    print("=" * 70)
    print("TASK 1: WEB SCRAPING — books.toscrape.com")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_books = []

    # Scrape catalogue pages (site has 50 pages, 20 books each = 1000 books)
    # We'll scrape first 5 pages for a good dataset (100 books)
    total_pages = 5
    print(f"\n📖 Scraping {total_pages} pages from books.toscrape.com...\n")

    scrape_success = False
    for page in range(1, total_pages + 1):
        print(f"  📄 Scraping page {page}/{total_pages}...")
        books = scrape_books_page(page)
        if not books:
            print(f"  ⚠ No books found on page {page}. Stopping.")
            break

        # Fetch category and description for each book
        for i, book in enumerate(books):
            details = scrape_book_details(book["Book_URL"])
            book.update(details)
            if (i + 1) % 5 == 0:
                print(f"    → Scraped {i + 1}/{len(books)} book details...")

        all_books.extend(books)
        scrape_success = True
        print(f"  ✓ Page {page}: {len(books)} books scraped")
        time.sleep(1)  # Be polite to the server

    # Use fallback if scraping failed
    if not scrape_success or len(all_books) == 0:
        df = offline_fallback()
        if df is None:
            print("  ✗ No data available. Exiting.")
            return None
    else:
        df = pd.DataFrame(all_books)
        # Reorder columns
        columns = ["Title", "Price", "Rating", "Rating_Numeric", "Availability",
                    "Category", "Description", "Book_URL"]
        df = df[columns]

    df.to_csv(OUTPUT_FILE, index=False)

    # Print summary
    print(f"\n{'─' * 70}")
    print(f"✅ SCRAPING COMPLETE!")
    print(f"{'─' * 70}")
    print(f"  Total books scraped : {len(df)}")
    print(f"  Columns             : {', '.join(df.columns)}")
    if "Category" in df.columns:
        print(f"  Unique categories   : {df['Category'].nunique()}")
    print(f"  Price range         : £{df['Price'].min():.2f} — £{df['Price'].max():.2f}")
    print(f"  Output saved to     : {OUTPUT_FILE}")
    print(f"{'─' * 70}\n")

    # Display sample
    print("📋 Sample Data (first 5 rows):")
    display_cols = [c for c in ["Title", "Price", "Rating", "Category"] if c in df.columns]
    print(df[display_cols].head().to_string(index=False))
    print()

    return df


if __name__ == "__main__":
    main()


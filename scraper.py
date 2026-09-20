import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://books.toscrape.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# STORAGE
# ============================================================

all_books = []

current_url = BASE_URL
page_number = 1


# ============================================================
# SCRAPE WEBSITE
# ============================================================

while current_url:

    print(f"Scraping page: {page_number}")

    try:

        # ----------------------------------------------------
        # REQUEST PAGE
        # ----------------------------------------------------

        response = requests.get(
            current_url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()


        # ----------------------------------------------------
        # PARSE HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        # ----------------------------------------------------
        # FIND BOOKS
        # ----------------------------------------------------

        books = soup.find_all(
            "article",
            class_="product_pod"
        )

        print(f"Books found: {len(books)}")


        # ----------------------------------------------------
        # EXTRACT EACH BOOK
        # ----------------------------------------------------

        for book in books:

            # =================================================
            # TITLE
            # =================================================

            title_tag = book.find("h3").find("a")

            title = title_tag.get(
                "title",
                ""
            ).strip()


            # =================================================
            # BOOK URL
            # =================================================

            book_url = title_tag.get(
                "href",
                ""
            )

            book_url = requests.compat.urljoin(
                current_url,
                book_url
            )


            # =================================================
            # PRICE
            # =================================================

            price_tag = book.find(
                "p",
                class_="price_color"
            )

            price_text = price_tag.get_text(
                strip=True
            )

            price_text = (
                price_text
                .replace("Â", "")
                .replace("£", "")
                .strip()
            )

            try:

                price = float(price_text)

            except ValueError:

                price = 0.0


            # =================================================
            # RATING
            # =================================================

            rating_tag = book.find(
                "p",
                class_="star-rating"
            )

            if rating_tag:

                rating_classes = rating_tag.get(
                    "class",
                    []
                )

                rating = next(
                    (
                        item
                        for item in rating_classes
                        if item != "star-rating"
                    ),
                    "Unknown"
                )

            else:

                rating = "Unknown"


            # =================================================
            # AVAILABILITY
            # =================================================

            availability_tag = book.find(
                "p",
                class_="instock availability"
            )

            if availability_tag:

                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

            else:

                availability = "Unknown"


            # =================================================
            # IMAGE URL
            # =================================================

            image_tag = book.find("img")

            if image_tag:

                image_url = image_tag.get(
                    "src",
                    ""
                )

                image_url = requests.compat.urljoin(
                    current_url,
                    image_url
                )

            else:

                image_url = ""


            # =================================================
            # CATEGORY
            # =================================================

            category = "Unknown"

            try:

                detail_response = requests.get(
                    book_url,
                    headers=HEADERS,
                    timeout=10
                )

                detail_response.raise_for_status()

                detail_soup = BeautifulSoup(
                    detail_response.text,
                    "html.parser"
                )

                breadcrumb = detail_soup.find(
                    "ul",
                    class_="breadcrumb"
                )

                if breadcrumb:

                    items = breadcrumb.find_all("li")

                    if len(items) >= 3:

                        category = items[2].get_text(
                            strip=True
                        )

            except requests.exceptions.RequestException:

                category = "Unknown"


            # =================================================
            # SAVE RECORD
            # =================================================

            all_books.append({

                "Title": title,

                "Price": price,

                "Rating": rating,

                "Availability": availability,

                "Book URL": book_url,

                "Image URL": image_url,

                "Category": category

            })


        # ----------------------------------------------------
        # FIND NEXT PAGE
        # ----------------------------------------------------

        next_button = soup.find(
            "li",
            class_="next"
        )

        if next_button:

            next_link = next_button.find("a")

            if next_link:

                next_url = next_link.get(
                    "href"
                )

                current_url = requests.compat.urljoin(
                    current_url,
                    next_url
                )

                page_number += 1

                time.sleep(0.3)

            else:

                current_url = None

        else:

            current_url = None


    except requests.exceptions.RequestException as error:

        print(
            f"Error on page {page_number}: {error}"
        )

        break


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_books)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(
    subset=["Book URL"],
    inplace=True
)


# ============================================================
# CLEAN DATA
# ============================================================

df["Title"] = df[
    "Title"
].fillna("Unknown")


df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
).fillna(0)


df["Rating"] = df[
    "Rating"
].fillna("Unknown")


df["Availability"] = df[
    "Availability"
].fillna("Unknown")


df["Book URL"] = df[
    "Book URL"
].fillna("")


df["Image URL"] = df[
    "Image URL"
].fillna("")


df["Category"] = df[
    "Category"
].fillna("Unknown")


# ============================================================
# CREATE DATA FOLDER
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    "data/books.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV file created successfully!")


# ============================================================
# SAVE JSON
# ============================================================

df.to_json(
    "data/books.json",
    orient="records",
    indent=4,
    force_ascii=False
)

print("JSON file created successfully!")


# ============================================================
# SAVE EXCEL
# ============================================================

df.to_excel(
    "data/books.xlsx",
    index=False
)

print("Excel file created successfully!")


# ============================================================
# DATA QUALITY REPORT
# ============================================================

print("\n========================================")
print("DATA QUALITY REPORT")
print("========================================")


print("\nMissing values:")

print(
    df.isnull().sum()
)


print("\nDuplicate records:")

print(
    df.duplicated().sum()
)


print("\nDataset information:")

df.info()


print("\nFirst 5 records:")

print(
    df.head()
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("SCRAPING COMPLETED!")
print("========================================")

print(
    f"Pages scraped: {page_number}"
)

print(
    f"Total books: {len(df)}"
)

print(
    "CSV: data/books.csv"
)

print(
    "JSON: data/books.json"
)

print(
    "Excel: data/books.xlsx"
)

print("========================================")
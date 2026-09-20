# Web Scraping Project

## Task 1: Web Scraping

### Objective

The objective of this task is to collect structured data from a public website using Python web scraping techniques.

The project demonstrates how to:

- Send HTTP requests to a website
- Understand HTML structure
- Extract data using BeautifulSoup
- Navigate through multiple webpages
- Clean the collected data
- Store data in CSV and JSON formats
- Check missing and duplicate values

## Website Used

Books to Scrape:

https://books.toscrape.com/

This website is used for practicing web scraping.

## Technologies Used

- Python
- Requests
- BeautifulSoup4
- Pandas
- VS Code

## Dataset Information

The scraper collects information about books from the website.

The dataset contains the following columns:

| Column | Description |
|---|---|
| Title | Name of the book |
| Price | Price of the book |
| Rating | Book rating |
| Availability | Availability status |
| Book URL | URL of the book |

## Web Scraping Process

The program performs the following steps:

1. Connects to the website using Requests.
2. Downloads the webpage HTML.
3. Parses the HTML using BeautifulSoup.
4. Finds book information using HTML tags and CSS classes.
5. Extracts title, price, rating, availability, and URL.
6. Navigates through five pages.
7. Collects approximately 100 book records.
8. Cleans the price values.
9. Checks missing values and duplicate records.
10. Saves the dataset as CSV and JSON.

## Output Files

### CSV

`books.csv`

Contains the scraped data in CSV format.

### JSON

`books.json`

Contains the same scraped data in JSON format.

## How to Run

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
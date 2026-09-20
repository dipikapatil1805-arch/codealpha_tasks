from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

DATABASE_PATH = "database/books.db"


def get_database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def home():

    connection = get_database_connection()

    # Total books
    total_books = connection.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    # Average price
    average_price = connection.execute(
        "SELECT AVG(Price) FROM books"
    ).fetchone()[0]

    # Books in stock
    in_stock = connection.execute(
        """
        SELECT COUNT(*)
        FROM books
        WHERE Availability LIKE '%In stock%'
        """
    ).fetchone()[0]

    # Number of different ratings
    rating_count = connection.execute(
        "SELECT COUNT(DISTINCT Rating) FROM books"
    ).fetchone()[0]

    connection.close()

    # Handle empty database
    if average_price is None:
        average_price = 0
    else:
        average_price = round(average_price, 2)

    return render_template(
        "index.html",
        total_books=total_books,
        average_price=average_price,
        in_stock=in_stock,
        rating_count=rating_count
    )


# =========================================================
# BOOKS PAGE
# =========================================================

@app.route("/books")
def books():

    # Get search value from URL
    search = request.args.get("search", "").strip()

    connection = get_database_connection()

    # If user searched for something
    if search:

        books = connection.execute(
            """
            SELECT Title, Price, Rating, Availability, [Book URL]
            FROM books
            WHERE Title LIKE ?
            ORDER BY Title
            """,
            ("%" + search + "%",)
        ).fetchall()

    # If no search value
    else:

        books = connection.execute(
            """
            SELECT Title, Price, Rating, Availability, [Book URL]
            FROM books
            ORDER BY Title
            """
        ).fetchall()

    connection.close()

    return render_template(
        "books.html",
        books=books,
        search=search
    )


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)


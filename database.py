import sqlite3
import os
import pandas as pd


DATABASE_PATH = "database/books.db"
CSV_PATH = "data/books.csv"


def create_database():

    os.makedirs("database", exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price REAL,
            rating TEXT,
            availability TEXT,
            book_url TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")


def insert_books():

    df = pd.read_csv(CSV_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    df.to_sql(
        "books",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    print("Books inserted successfully!")
    print("Total books:", len(df))


if __name__ == "__main__":

    create_database()

    insert_books()
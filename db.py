import sqlite3

DB_PATH = "meetings.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            transcript TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_meeting(title: str, transcript: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO meetings (title, transcript) VALUES (?, ?)",
        (title, transcript)
    )

    conn.commit()
    conn.close()


def load_meeting_transcript(title: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT transcript FROM meetings WHERE title = ?", (title,))
    row = cursor.fetchone()

    conn.close()

    return row[0] if row else None

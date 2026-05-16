import sqlite3
from pathlib import Path

db_path = Path("database/fleea.db")

def migrate():
    if not db_path.exists():
        print("Database not found.")
        return
        
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        conn.commit()
        print("Successfully added role column.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()

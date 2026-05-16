import sys
import os
import bcrypt
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from database.models import DatabaseManager
from config.settings import Settings

def create_admin():
    settings = Settings()
    db = DatabaseManager(settings.db_path_resolved)
    
    user_id = "mantu2131"
    password = "mani@mantu"
    name = "Mantu (Admin)"
    email = "admin@mantu.com"
    
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    
    with db.connection() as conn:
        try:
            conn.execute(
                "INSERT INTO users (user_id, name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, email, hashed_pw, "admin", "2026-05-16T17:07:07Z")
            )
            print(f"Admin user '{user_id}' created successfully.")
        except Exception as e:
            print(f"Error creating admin: {e}")

if __name__ == "__main__":
    create_admin()

import psycopg2
from app.config import get_settings

def create_table():
    settings = get_settings()
    if not settings.database_url:
        print("❌ DATABASE_URL not set in config.")
        return

    try:
        conn = psycopg2.connect(settings.database_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oauthtokens (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(50) UNIQUE NOT NULL,
                    access_token TEXT,
                    refresh_token TEXT,
                    expires_in FLOAT,
                    obtained_at FLOAT
                );
            """)
            print("✅ Table 'oauthtokens' created successfully.")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_table()

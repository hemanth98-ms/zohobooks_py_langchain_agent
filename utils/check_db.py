import psycopg2
from app.config import get_settings

def check():
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    with conn.cursor() as cur:
        # Check table existence
        cur.execute("SELECT to_regclass('public.invoice_embeddings');")
        exists = cur.fetchone()[0]
        
        if not exists:
            print("❌ Table 'invoice_embeddings' does NOT exist.")
        else:
            print("✅ Table 'invoice_embeddings' exists.")
            # Check count
            cur.execute("SELECT count(*) FROM invoice_embeddings;")
            count = cur.fetchone()[0]
            print(f"📊 Total Rows: {count}")
            
            if count > 0:
                cur.execute("SELECT invoice_id, content FROM invoice_embeddings LIMIT 1;")
                row = cur.fetchone()
                print(f"👀 Sample: {row}")

if __name__ == "__main__":
    check()

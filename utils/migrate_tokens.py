from app.zoho_books import ZohoOAuthClient
import json
import os

def migrate():
    print("🚀 Starting Token Migration...")
    client = ZohoOAuthClient()
    
    # Force load from file if not loaded (the class init tries DB first)
    if not client.tokens and os.path.isfile(client.token_path):
        print("   Reading from local JSON file...")
        with open(client.token_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print("   Saving to Database...")
        # This will write to DB because we updated the method
        client._save_tokens(data)
        print("✅ Migration Successful! Tokens are now in Neon DB.")
    elif client.tokens:
        print("ℹ️ Tokens already loaded (possibly from DB or File).")
        # Force save to ensure DB has it
        data = {
            "access_token": client.tokens.access_token,
            "refresh_token": client.tokens.refresh_token,
            "expires_in": client.tokens.expires_in,
            "obtained_at": client.tokens.obtained_at
        }
        client._save_tokens(data)
        print("✅ Synced in-memory tokens to DB.")
    else:
        print("❌ No tokens found locally to migrate.")

if __name__ == "__main__":
    migrate()

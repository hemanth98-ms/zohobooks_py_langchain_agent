"""
Zoho OAuth Diagnostic Tool
Tests different Zoho data centers and validates credentials
"""
import requests
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("ZOHO_REDIRECT_URI")

# Different Zoho data centers
DATA_CENTERS = {
    "US": "https://accounts.zoho.com",
    "EU": "https://accounts.zoho.eu",
    "IN": "https://accounts.zoho.in",
    "AU": "https://accounts.zoho.com.au",
    "CN": "https://accounts.zoho.com.cn",
}

print("="*70)
print("ZOHO OAUTH DIAGNOSTIC TOOL")
print("="*70)
print(f"\nClient ID: {CLIENT_ID}")
print(f"Client Secret: {CLIENT_SECRET[:10]}..." if CLIENT_SECRET else "None")
print(f"Redirect URI: {REDIRECT_URI}")
print("\n" + "="*70)

def test_authorization_url(dc_name, base_url):
    """Generate authorization URL for each data center"""
    from urllib.parse import urlencode
    
    params = {
        "scope": "ZohoBooks.fullaccess.all",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "access_type": "offline",
        "redirect_uri": REDIRECT_URI,
        "prompt": "consent"
    }
    
    auth_url = f"{base_url}/oauth/v2/auth?{urlencode(params)}"
    print(f"\n{dc_name} Data Center:")
    print(f"  {auth_url}")

print("\nAUTHORIZATION URLs FOR DIFFERENT DATA CENTERS:")
print("-"*70)
for dc_name, base_url in DATA_CENTERS.items():
    test_authorization_url(dc_name, base_url)

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("""
1. Try the authorization URL for YOUR data center (usually US/IN/EU)
2. After authorizing, you'll get a code in the redirect URL
3. Run this script with the code to test token exchange:
   
   python test_zoho_oauth.py YOUR_CODE_HERE

4. If you still get 'invalid_client', verify in Zoho API Console:
   - Go to: https://api-console.zoho.com/
   - Check your Client ID and Secret match exactly
   - Verify the Redirect URI is: http://localhost:8000/callback
   - Make sure the client is for the correct data center
""")

# If code is provided as argument, test token exchange
import sys
if len(sys.argv) > 1:
    code = sys.argv[1]
    print("\n" + "="*70)
    print("TESTING TOKEN EXCHANGE WITH ALL DATA CENTERS")
    print("="*70)
    
    for dc_name, base_url in DATA_CENTERS.items():
        print(f"\nTrying {dc_name} ({base_url})...")
        
        url = f"{base_url}/oauth/v2/token"
        params = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "code": code,
        }
        
        try:
            response = requests.post(url, params=params, timeout=30)
            
            if response.status_code == 200:
                print(f"  ✅ SUCCESS with {dc_name}!")
                token_data = response.json()
                print(f"\n  Access Token: {token_data.get('access_token')}")
                print(f"  Refresh Token: {token_data.get('refresh_token')}")
                
                # Save tokens
                import json
                import time
                os.makedirs("db", exist_ok=True)
                save_data = {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_in": token_data.get("expires_in", 3600),
                    "obtained_at": time.time(),
                    "data_center": dc_name,
                    "accounts_base": base_url
                }
                
                with open("db/zoho_tokens.json", "w") as f:
                    json.dump(save_data, f, indent=2)
                
                print(f"\n  ✅ Tokens saved to db/zoho_tokens.json")
                print(f"\n  Update your .env with:")
                print(f"  ZOHO_ACCOUNTS_BASE={base_url}")
                break
            else:
                print(f"  ❌ Failed: {response.json()}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

# Database-Only Setup (No Zoho OAuth Required!)

## What This Version Does

✅ **Works without Zoho authentication**  
✅ **Queries invoice data from your database**  
✅ **Answers general questions**  
❌ **Cannot fetch live data from Zoho Books API**

## Requirements

1. **Database with invoice data** - You need PostgreSQL with invoice data already loaded
2. **API Keys** - Groq and Google AI (for LLM and embeddings)
3. **No Zoho OAuth needed!**

## Setup

### 1. Configure `.env`

```env
# Required
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql://user:password@host:port/database

# Optional
LLM_MODEL=llama-3.1-8b-instant
EMBEDDINGS_MODEL=models/text-embedding-004
```

### 2. Make sure database has invoice data

If you don't have data yet, use the full version to load it:

```bash
python zoho_books_all_in_one.py  # This needs Zoho OAuth
python utils/ingest.py            # Load invoices into database
```

### 3. Run the database-only version

```bash
python zoho_books_db_only.py
```

## Usage

**Questions it can answer:**

- "Search for invoices containing laptops"
- "Find invoices for customer John"
- "Show me invoices over $1000"
- "What is an invoice?" (general question)

**Questions it CANNOT answer:**

- "List all invoices" (needs live API)
- "Create a refund" (needs live API)
- "Get latest invoice" (needs live API)

## Perfect For

- **Sharing with friends** - No complex OAuth setup
- **Demo purposes** - Works with sample data
- **Offline queries** - Database-only, no API calls
- **Testing** - Quick setup without authentication

## Limitations

- Only searches data already in the database
- Cannot fetch new/updated invoices from Zoho
- Cannot create or modify invoices
- Requires database to be populated first

---

**TL;DR:** Use this version if you want to share with someone who doesn't have Zoho Books access!

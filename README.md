# Zoho Books Agent

A conversational AI agent for Zoho Books that can answer both billing-related questions and general queries.

## 📁 Project Structure

```
zoho-books-demo/
├── zoho_books_all_in_one.py    # 🎯 Main agent (all-in-one file)
├── chat_cli.py                  # 💬 Interactive CLI chat
├── server.py                    # 🌐 Web server (FastAPI)
├── requirements.txt             # 📦 Python dependencies
├── .env                         # 🔐 Configuration (not in git)
│
├── app/
│   └── static/
│       └── index.html          # 🎨 Web chat interface
│
├── db/
│   └── zoho_tokens.json        # 🔑 OAuth tokens (not in git)
│
└── utils/                       # 🛠️ One-time setup utilities
    ├── test_zoho_oauth.py      # OAuth testing
    ├── check_db.py             # Database verification
    ├── create_token_table.py   # DB table creation
    ├── migrate_tokens.py       # Token migration
    └── ingest.py               # RAG data ingestion
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=your_postgres_connection_string
ZOHO_CLIENT_ID=your_zoho_client_id
ZOHO_CLIENT_SECRET=your_zoho_client_secret
ZOHO_ORG_ID=your_zoho_org_id
```

### 3. Run the Agent

**Option A: Interactive CLI**

```bash
python chat_cli.py
```

**Option B: Web Interface**

```bash
python server.py
```

Then open http://localhost:8000 in your browser.

## 💡 Usage Examples

### Zoho Books Questions

- "List all my invoices"
- "Find customer with email john@example.com"
- "Show me invoice INV-000056"
- "Search for invoices containing laptops"

### General Questions

- "What is an invoice?"
- "Explain Zoho Books"
- "What is Python?"
- "How does OAuth work?"

## 🛠️ Utilities (utils/)

These are one-time setup scripts:

- **test_zoho_oauth.py** - Test OAuth configuration
- **create_token_table.py** - Create database table for tokens
- **migrate_tokens.py** - Migrate tokens from file to database
- **check_db.py** - Verify database connection and data
- **ingest.py** - Load invoice data into RAG system

## 📝 Features

✅ Conversational AI with Groq LLM  
✅ Zoho Books API integration  
✅ OAuth 2.0 authentication  
✅ Vector search (RAG) for invoices  
✅ Web and CLI interfaces  
✅ Handles both specific and general questions

## 🔧 Tech Stack

- **LLM**: Groq (Llama 3.1)
- **Framework**: LangChain
- **Web**: FastAPI
- **Database**: PostgreSQL with pgvector
- **Embeddings**: Google Generative AI

## 📄 License

MIT

"""
Zoho Books Agent - All-in-One Consolidated File
Combines all functionality: Config, Base Classes, OAuth, RAG, Tools, and Agent
"""

import os
import json
import time
import psycopg2
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

class Settings(BaseModel):
    app_title: str = os.getenv("APP_TITLE", "Multi-Agent Support API")
    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM/Embeddings
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")

    # Vector store
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", ".vector_store")

    # Databases
    crm_db_path: str = os.getenv("CRM_DB_PATH", "db/crm.sqlite")
    orders_db_path: str = os.getenv("ORDERS_DB_PATH", "db/orders.sqlite")
    database_url: str | None = os.getenv("DATABASE_URL")

    # Zoho Books OAuth
    zoho_client_id: str | None = os.getenv("ZOHO_CLIENT_ID")
    zoho_client_secret: str | None = os.getenv("ZOHO_CLIENT_SECRET")
    zoho_redirect_uri: str | None = os.getenv("ZOHO_REDIRECT_URI")
    zoho_org_id: str | None = os.getenv("ZOHO_ORG_ID")
    zoho_token_path: str = os.getenv("ZOHO_TOKEN_PATH", "db/zoho_tokens.json")
    zoho_accounts_base: str = os.getenv("ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.in")
    zoho_books_base: str = os.getenv("ZOHO_BOOKS_BASE", "https://www.zohoapis.in/books/v3")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ============================================================================
# BASE CLASSES
# ============================================================================

class Agent:
    def __init__(self):
        pass


@dataclass
class AgentResult:
    text: str
    steps: List[str]
    confidence: float
    agent_name: str


@dataclass
class OAuthTokens:
    access_token: str
    refresh_token: Optional[str]
    expires_in: int
    obtained_at: float

    @property
    def expires_at(self) -> float:
        return self.obtained_at + max(0, self.expires_in - 30)  # refresh 30s early

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


# ============================================================================
# RAG SERVICE
# ============================================================================

class RAGService:
    def __init__(self):
        self.settings = get_settings()
        if not self.settings.database_url:
            raise ValueError("DATABASE_URL is not set.")
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        genai.configure(api_key=self.settings.google_api_key)
        self.embedding_model = self.settings.embeddings_model
        
        self.conn = psycopg2.connect(self.settings.database_url)
        self.conn.autocommit = True
        self.init_db()

    def init_db(self):
        """Initializes the vector extension and table."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS invoice_embeddings (
                    id SERIAL PRIMARY KEY,
                    invoice_id VARCHAR(50) UNIQUE,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(768)
                );
            """)

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single text."""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=768
        )
        return result["embedding"]

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a query."""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        return result["embedding"]

    def add_invoice(self, invoice_data: Dict):
        """Adds or updates an invoice in the vector store."""
        invoice_id = invoice_data.get("invoice_id")
        if not invoice_id:
            return

        lines = []
        lines.append(f"Invoice #{invoice_data.get('invoice_number')} for {invoice_data.get('customer_name')}")
        lines.append(f"Date: {invoice_data.get('date')}, Status: {invoice_data.get('status')}")
        lines.append(f"Total: {invoice_data.get('total')}, Balance: {invoice_data.get('balance')}")
        
        if "line_items" in invoice_data:
            lines.append("Items:")
            for item in invoice_data["line_items"]:
                lines.append(f"- {item.get('name')} (Qty: {item.get('quantity')}, Rate: {item.get('rate')})")
        
        content = "\n".join(lines)
        embedding = self.embed_text(content)
        metadata_json = json.dumps(invoice_data)

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO invoice_embeddings (invoice_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (invoice_id) 
                DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding;
            """, (str(invoice_id), content, metadata_json, embedding))

    def search_invoices(self, query: str, limit: int = 5) -> List[Dict]:
        """Searches for similar invoices."""
        # Validate query is not empty
        if not query or not query.strip():
            return [{
                "content": "No search query provided",
                "metadata": {},
                "similarity": 0.0
            }]
        
        query_embedding = self.embed_query(query.strip())
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM invoice_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, query_embedding, limit))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "content": row[0],
                    "metadata": row[1],
                    "similarity": row[2]
                })
            return results


# ============================================================================
# ZOHO OAUTH CLIENT
# ============================================================================

class ZohoOAuthClient:
    def __init__(self):
        self.settings = get_settings()
        self.tokens: Optional[OAuthTokens] = None
        self.conn = None
        
        # Init DB connection
        if self.settings.database_url:
            try:
                self.conn = psycopg2.connect(self.settings.database_url)
                self.conn.autocommit = True
            except Exception as e:
                print(f"⚠️ Could not connect to DB for tokens: {e}")

        self._load_tokens()

    @property
    def accounts_base(self) -> str:
        return self.settings.zoho_accounts_base.rstrip("/")

    @property
    def books_base(self) -> str:
        return self.settings.zoho_books_base.rstrip("/")

    @property
    def org_id(self) -> Optional[str]:
        return self.settings.zoho_org_id

    @property
    def token_path(self) -> str:
        return self.settings.zoho_token_path

    def _load_tokens(self):
        # Try Loading from DB first
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT access_token, refresh_token, expires_in, obtained_at FROM oauthtokens WHERE service_name = 'zoho_books'")
                    row = cur.fetchone()
                    if row:
                        self.tokens = OAuthTokens(
                            access_token=row[0],
                            refresh_token=row[1],
                            expires_in=int(row[2]),
                            obtained_at=float(row[3])
                        )
                        print("✅ Loaded tokens from Database")
                        return
            except Exception as e:
                print(f"⚠️ Error loading tokens from DB: {e}")

        # Fallback to File
        if os.path.isfile(self.token_path):
            try:
                with open(self.token_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tokens = OAuthTokens(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_in=int(data.get("expires_in", 3600)),
                    obtained_at=float(data.get("obtained_at", time.time())),
                )
                print("✅ Loaded tokens from File")
            except Exception:
                self.tokens = None
        else:
            print("ℹ️ No tokens found (DB or File). Auth required.")

    def _save_tokens(self, data: dict):
        payload = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token") or (self.tokens.refresh_token if self.tokens else None),
            "expires_in": int(data.get("expires_in", 3600)),
            "obtained_at": time.time(),
        }

        # Update In-Memory
        self.tokens = OAuthTokens(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_in=payload["expires_in"],
            obtained_at=payload["obtained_at"],
        )

        # Save to DB
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO oauthtokens (service_name, access_token, refresh_token, expires_in, obtained_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (service_name)
                        DO UPDATE SET access_token = EXCLUDED.access_token, 
                                      refresh_token = COALESCE(EXCLUDED.refresh_token, oauthtokens.refresh_token),
                                      expires_in = EXCLUDED.expires_in,
                                      obtained_at = EXCLUDED.obtained_at;
                    """, ('zoho_books', payload['access_token'], payload['refresh_token'], payload['expires_in'], payload['obtained_at']))
                print("✅ Saved tokens to Database")
            except Exception as e:
                print(f"❌ Failed to save tokens to DB: {e}")

        # Save to File (Backup)
        try:
            os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
            with open(self.token_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print("✅ Saved tokens to File")
        except Exception as e:
            print(f"⚠️ Failed to save tokens to File: {e}")

    def exchange_code(self, code: str) -> None:
        url = f"{self.accounts_base}/oauth/v2/token"
        params = {
            "grant_type": "authorization_code",
            "client_id": self.settings.zoho_client_id,
            "client_secret": self.settings.zoho_client_secret,
            "redirect_uri": self.settings.zoho_redirect_uri,
            "code": code,
        }
        r = requests.post(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._save_tokens(data)

    def _refresh(self) -> None:
        if not (self.tokens and self.tokens.refresh_token):
            raise RuntimeError("No refresh_token available. Complete OAuth first.")
        url = f"{self.accounts_base}/oauth/v2/token"
        params = {
            "grant_type": "refresh_token",
            "client_id": self.settings.zoho_client_id,
            "client_secret": self.settings.zoho_client_secret,
            "refresh_token": self.tokens.refresh_token,
        }
        r = requests.post(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._save_tokens(data)

    def get_access_token(self) -> str:
        if not self.tokens:
            raise RuntimeError("Zoho OAuth tokens not found. Use exchange_code() to initialize.")
        if self.tokens.is_expired():
            self._refresh()
        return self.tokens.access_token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Zoho-oauthtoken {self.get_access_token()}",
            "X-com-zoho-books-organizationid": self.org_id or "",
            "Content-Type": "application/json",
        }

    # Basic helpers for common Books endpoints
    def list_invoices(self, page: int = 1, per_page: int = 10, invoice_number: Optional[str] = None) -> dict:
        url = f"{self.books_base}/invoices"
        params = {"page": page, "per_page": per_page}
        if invoice_number:
            params["invoice_number_contains"] = invoice_number
        r = requests.get(url, headers=self._headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_invoice(self, invoice_id: str) -> dict:
        url = f"{self.books_base}/invoices/{invoice_id}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_customer(self, customer_id: str) -> dict:
        url = f"{self.books_base}/api/v3/contacts/{customer_id}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def search_customer(self, email: Optional[str] = None, name: Optional[str] = None, page: int = 1) -> dict:
        url = f"{self.books_base}/api/v3/contacts"
        params: Dict[str, str | int] = {"page": page}
        if email:
            params["email_contains"] = email
        if name:
            params["name_contains"] = name
        r = requests.get(url, headers=self._headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_subscriptions(self, page: int = 1, per_page: int = 10) -> dict:
        url = f"{self.books_base}/api/v3/subscriptions"
        params = {"page": page, "per_page": per_page}
        r = requests.get(url, headers=self._headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def create_refund(self, invoice_id: str, amount: float, date: str) -> dict:
        url = f"{self.books_base}/api/v3/customerpayments/refunds"
        payload = {"invoice_id": invoice_id, "amount": amount, "date": date}
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()


# ============================================================================
# TOOL SCHEMAS
# ============================================================================

class ListInvoicesArgs(BaseModel):
    page: int = Field(1, description="Page number for pagination")
    per_page: int = Field(5, description="Number of invoices per page (default 5, max 10)")
    invoice_number: Optional[str] = Field(None, description="Filter by invoice number")


class GetInvoiceArgs(BaseModel):
    invoice_id: str = Field(..., description="The ID (not number) of the invoice")


class SearchCustomerArgs(BaseModel):
    name: Optional[str] = Field(None, description="Name of the customer to search")
    email: Optional[str] = Field(None, description="Email of the customer to search")
    page: int = Field(1, description="Page number")


class CreateRefundArgs(BaseModel):
    invoice_id: str = Field(..., description="ID of the invoice to refund")
    amount: float = Field(..., description="Amount (value) to refund")
    date: str = Field(..., description="Date of refund in YYYY-MM-DD format")


class SearchContentArgs(BaseModel):
    query: str = Field(..., description="The search query (e.g., 'adobe subscription', 'laptop')")
    limit: int = Field(5, description="Number of results to return")


# ============================================================================
# TOOL SET
# ============================================================================

class ZohoToolSet:
    def __init__(self, client, rag_service=None):
        self.client = client
        self.rag = rag_service

    def get_tools(self) -> List[StructuredTool]:
        tools = [
            StructuredTool.from_function(
                func=self.client.list_invoices,
                name="list_invoices",
                description="List invoices from Zoho Books. Returns a list of invoice summaries.",
                args_schema=ListInvoicesArgs
            ),
            StructuredTool.from_function(
                func=self.client.get_invoice,
                name="get_invoice",
                description="Get full details of a specific invoice using its invoice_id.",
                args_schema=GetInvoiceArgs
            ),
            StructuredTool.from_function(
                func=self.client.search_customer,
                name="search_customer",
                description="Search for a customer by name or email.",
                args_schema=SearchCustomerArgs
            ),
            StructuredTool.from_function(
                func=self.client.create_refund,
                name="create_refund",
                description="Create a refund for a specific invoice.",
                args_schema=CreateRefundArgs
            )
        ]
        
        if self.rag:
            tools.append(
                StructuredTool.from_function(
                    func=self.rag.search_invoices,
                    name="search_invoices",
                    description="Semantic search over invoice content (items, descriptions). Use this for broad queries like 'What did I buy from Adobe?'",
                    args_schema=SearchContentArgs
                )
            )
            
        return tools


# ============================================================================
# ZOHO BOOKS AGENT
# ============================================================================

class ZohoBooksAgent(Agent):
    name = "billing"
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.client = ZohoOAuthClient()
        
        # Initialize RAG
        try:
            self.rag = RAGService()
        except Exception as e:
            print(f"Warning: RAG not initialized: {e}")
            self.rag = None

        # Verify API key
        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        # Initialize LLM
        self.llm = ChatGroq(
            api_key=self.settings.groq_api_key,
            model_name=self.settings.llm_model,
            temperature=0
        )

        # Initialize Tools
        toolset = ZohoToolSet(client=self.client, rag_service=self.rag)
        self.tools = toolset.get_tools()

        # Initialize Agent
        system_prompt = """You are a helpful AI assistant with expertise in Zoho Books billing and accounting.

For Zoho Books questions:
- Use the provided tools to list invoices, search customers, create refunds, get invoice details, and search document content
- If a user asks for invoice details, find it first via search or list, then get full details
- Use 'search_invoices' for broad queries involving items or products

For general questions:
- Answer directly using your knowledge without using tools
- Be helpful, accurate, and concise
- If you're not sure about something, say so

Determine whether the question is about Zoho Books (use tools) or a general question (answer directly)."""

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            state_schema=AgentState,
            system_prompt=system_prompt
        )

    def _need_oauth_instructions(self) -> AgentResult:
        auth_url = (
            f"{self.client.accounts_base}/oauth/v2/auth?"
            f"scope=ZohoBooks.fullaccess.all&"
            f"client_id={self.client.settings.zoho_client_id}&"
            f"response_type=code&access_type=offline&"
            f"redirect_uri={self.client.settings.zoho_redirect_uri}&"
            f"prompt=consent"
        )
        text = (
            "Zoho Books isn't authorized yet. Please complete OAuth:\n"
            f"1) Visit: {auth_url}\n"
            "2) Sign in and approve.\n"
            "3) Capture the 'code' from redirect URL.\n"
            "4) Call POST /admin/zoho/exchange?code=YOUR_CODE to store tokens.\n"
        )
        return AgentResult(text=text, steps=["zoho.oauth_required"], confidence=0.2, agent_name=self.name)

    def run(self, message: str, context: Dict[str, Any]) -> AgentResult:
        # Check OAuth First
        try:
            _ = self.client.get_access_token()
        except RuntimeError:
            return self._need_oauth_instructions()
        except Exception:
            return self._need_oauth_instructions()

        # Prepare History
        raw_history = context.get("history", [])
        chat_history = []
        for msg in raw_history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                chat_history.append(AIMessage(content=msg.get("content")))

        # Run Agent
        try:
            result = self.agent.invoke({
                "messages": [HumanMessage(content=message)]
            })
            
            if "messages" in result and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                output_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                output_text = str(result)
            
            steps = [] 

            return AgentResult(text=output_text, steps=steps, confidence=0.9, agent_name=self.name)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AgentResult(text=f"I encountered an error: {e}", steps=[], confidence=0.1, agent_name=self.name)


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Zoho Books Agent - All-in-One Version")
    print("=" * 60)
    
    # Example: Initialize the agent
    try:
        agent = ZohoBooksAgent()
        print("✅ Agent initialized successfully!")
        
        # Example query
        test_message = "List the first 3 invoices"
        test_context = {"history": []}
        
        result = agent.run(test_message, test_context)
        print(f"\n📝 Query: {test_message}")
        print(f"📤 Response: {result.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

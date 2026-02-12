"""
FastAPI Server for Database-Only Zoho Books Agent
Deploy this to Render.com for a live web interface
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from zoho_books_db_only import DatabaseOnlyAgent

app = FastAPI(title="Zoho Books Agent")

# Initialize agent once at startup
agent = None

@app.on_event("startup")
def startup():
    global agent
    try:
        agent = DatabaseOnlyAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

@app.post("/chat")
async def chat(request: Request):
    global agent
    if not agent:
        return JSONResponse({"error": "Agent not initialized"}, status_code=500)
    
    data = await request.json()
    message = data.get("message", "").strip()
    history = data.get("history", [])
    
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    
    result = agent.run(message, {"history": history})
    
    return {
        "text": result.text,
        "token_usage": result.token_usage,
        "confidence": result.confidence
    }

@app.get("/health")
def health():
    return {"status": "ok", "agent_ready": agent is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

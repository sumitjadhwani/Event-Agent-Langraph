"""FastAPI server exposing the LangGraph-based Event RAG chatbot."""

from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.messages import BaseMessage
from event_rag_agent.main import run_agent


STATIC_DIR = Path(__file__).parent / "static"
MAX_HISTORY_MESSAGES = 20
chat_sessions: Dict[str, List[BaseMessage]] = {}

app = FastAPI(
    title="Event RAG Agent",
    description="FastAPI + LangGraph chatbot for event information",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DocumentSnippet(BaseModel):
    title: str
    preview: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    used_retrieval: bool
    documents: List[DocumentSnippet]
    session_id: str


@app.get("/", response_class=FileResponse)
def read_root() -> FileResponse:
    """Serve the basic HTML UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="UI not found. Did you remove static/index.html?")
    return FileResponse(index_file)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the chatbot for the incoming message."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid4())
    history = chat_sessions.get(session_id, [])

    try:
        result = run_agent(message, history=history)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=f"Failed to run agent: {exc}") from exc

    raw_documents = result.get("documents") or []
    snippets: List[DocumentSnippet] = []
    for doc in raw_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        title = metadata.get("title") or metadata.get("event_name") or metadata.get("source") or "Event Detail"
        preview = (doc.page_content or "").strip().replace("\n", " ")[:200]
        snippets.append(DocumentSnippet(title=title, preview=preview))

    response_message = result["messages"][-1].content if result.get("messages") else "No response generated."
    used_retrieval = bool(result.get("should_retrieve"))

    updated_history = list(result.get("messages", history))
    if len(updated_history) > MAX_HISTORY_MESSAGES:
        updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
    chat_sessions[session_id] = updated_history

    return ChatResponse(
        response=response_message,
        used_retrieval=used_retrieval,
        documents=snippets,
        session_id=session_id,
    )


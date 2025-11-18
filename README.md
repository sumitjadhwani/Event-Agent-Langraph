## Event RAG Agent

LangGraph-based Retrieval-Augmented Generation chatbot for answering event related queries.

### Running the CLI demo

```bash
uv run python -m event_rag_agent.main
```

### Running the FastAPI + JS UI

1. Install dependencies (once):

```bash
uv sync
```

2. Start the FastAPI server:

```bash
uv run uvicorn server:app --reload
```

3. Open a browser at `http://127.0.0.1:8000` and chat with the bot.

### User preferences

A single `event_rag_agent/user_preferences.json` file keeps lightweight details (city, budget, interests). The agent tries to infer these from your messages and reuses them in future replies. In the future you can swap this for an authenticated profile store.

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

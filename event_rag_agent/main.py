"""Main application for Event RAG Agent using LangGraph."""

import json
import os
import re
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .config import (
    GROQ_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_DOCUMENTS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)
from .data import EVENTS


# ==================== INITIALIZE COMPONENTS ====================

# Set Groq API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

# Initialize LLM
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    timeout=LLM_TIMEOUT
)


# ==================== VECTOR STORE SETUP ====================

def setup_vectorstore():
    """Create or load the FAISS vector store."""
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing embeddings from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Embeddings loaded from disk")
    else:
        print("Creating new embeddings (this may take a moment)...")
        splits = text_splitter.split_documents(EVENTS)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"✓ Embeddings created and saved to {FAISS_INDEX_PATH}")
    
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_DOCUMENTS})


retriever = setup_vectorstore()


# ==================== SIMPLE PREFERENCE STORAGE ====================

PREFERENCES_PATH = Path(__file__).parent / "user_preferences.json"
DEFAULT_PREFERENCES: Dict[str, Any] = {
    "city": None,
    "budget": None,
    "interests": [],
}


def load_preferences() -> Dict[str, Any]:
    if not PREFERENCES_PATH.exists():
        save_preferences(DEFAULT_PREFERENCES)
    try:
        data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_PREFERENCES, **data}
    except json.JSONDecodeError:
        return DEFAULT_PREFERENCES.copy()


def save_preferences(preferences: Dict[str, Any]) -> None:
    PREFERENCES_PATH.write_text(
        json.dumps(preferences, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


# ==================== AGENT STATE ====================

class AgentState(TypedDict):
    """State schema for the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list
    query: str
    should_retrieve: bool
    needs_clarification: bool
    clarification_attempts: int
    preferences: Dict[str, Any]


# ==================== PREFERENCE HELPERS ====================

def update_preferences_from_text(text: str, current: Dict[str, Any]) -> Dict[str, Any]:
    """Very small heuristic layer to capture mentions of city, budget, or interests."""
    updated = current.copy()
    normalized = text.lower()

    city_match = re.search(r"(?:my city is|i'?m in|i am in|based in|city is)\s+([a-z\s]+)", normalized)
    if city_match:
        city = city_match.group(1).strip().title()
        updated["city"] = city

    budget_match = re.search(r"(?:budget|under|around)\s*[^\d]*(\d{3,6})", normalized)
    if budget_match:
        updated["budget"] = budget_match.group(1)

    for trigger in ("i like", "i love", "i enjoy"):
        if trigger in normalized:
            interest_segment = normalized.split(trigger, 1)[1]
            interest_segment = re.split(r"[.!?]", interest_segment)[0]
            interests = [part.strip().title() for part in re.split(r",|and", interest_segment) if part.strip()]
            if interests:
                deduped = list(dict.fromkeys(updated.get("interests", []) + interests))
                updated["interests"] = deduped
            break

    return updated


@tool("update_user_preferences")
def update_preferences_tool(
    text: str,
    current_preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract and persist city, budget, or interests mentioned in the text."""
    existing = current_preferences or load_preferences()
    updated = update_preferences_from_text(text, existing)
    changed = updated != existing
    if changed:
        save_preferences(updated)
    return {"preferences": updated, "changed": changed}


PREFERENCE_SYSTEM_PROMPT = (
    "You review the most recent user message and determine whether it contains new "
    "information about their preferred city, budget, or interests.\n"
    "- If it does, call the `update_user_preferences` tool exactly once, passing the "
    "raw user text and the current preferences.\n"
    "- If there is no new information, respond with 'No preference update needed.'"
)

preference_tools = [update_preferences_tool]
preference_tool_map = {tool.name: tool for tool in preference_tools}
preference_llm = llm.bind_tools(preference_tools)


def looks_like_error_page(content: str) -> bool:
    """Detect whether the LLM output appears to be an HTML error page."""
    if not content:
        return False

    probe = content.strip().lower()
    html_indicators = ("<!doctype", "<html", "<head", "<body")
    if any(probe.startswith(tag) for tag in html_indicators):
        return True
    return "cloudflare" in probe or "<script" in probe


def format_preferences(preferences: Dict[str, Any]) -> str:
    segments = []
    if preferences.get("city"):
        segments.append(f"Preferred city: {preferences['city']}")
    if preferences.get("budget"):
        segments.append(f"Budget: ₹{preferences['budget']}")
    if preferences.get("interests"):
        segments.append("Interests: " + ", ".join(preferences["interests"]))
    return "\n".join(segments)


# ==================== AGENT NODES ====================

def maybe_update_preferences(state: AgentState) -> dict:
    """Let the LLM decide whether to call the preference-update tool."""
    latest_user_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None,
    )
    if latest_user_message is None:
        return {}
    current_preferences = state.get("preferences") or DEFAULT_PREFERENCES.copy()

    try:
        decision = preference_llm.invoke(
            [
                SystemMessage(content=PREFERENCE_SYSTEM_PROMPT),
                HumanMessage(content=latest_user_message.content),
            ]
        )
    except Exception as exc:
        print(f"[PREFERENCES] Preference LLM failed: {exc}")
        return {}

    tool_calls = getattr(decision, "tool_calls", None)
    if not tool_calls:
        return {}

    updated_preferences = current_preferences

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        selected_tool = preference_tool_map.get(tool_name)
        if selected_tool is None:
            continue

        tool_args = dict(tool_call.get("args") or {})
        tool_args.setdefault("text", latest_user_message.content)
        tool_args.setdefault("current_preferences", updated_preferences)

        try:
            result = selected_tool.invoke(tool_args)
        except Exception as exc:
            print(f"[PREFERENCES] Tool '{tool_name}' failed: {exc}")
            continue

        if isinstance(result, dict) and "preferences" in result:
            updated_preferences = result["preferences"]

    if updated_preferences != state.get("preferences"):
        print("[PREFERENCES] Updated preferences via tool call.")
        return {"preferences": updated_preferences}

    return {}


def decide_to_retrieve(state: AgentState) -> dict:
    """Analyze query to decide if retrieval is needed or if a clarification loop is required."""
    query = state["messages"][-1].content
    query_lower = query.lower().strip()
    words = query_lower.split()

    knowledge_keywords = [
        "what", "how", "explain", "describe", "tell me about",
        "define", "who is", "why", "when", "where", "which",
        "event", "show", "find", "search", "list", "price",
        "concert", "workshop", "conference", "meetup",
    ]

    ambiguous_phrases = {
        "event",
        "events",
        "tell me more",
        "tell me something",
        "more details",
        "help",
        "info",
    }

    should_retrieve = any(keyword in query_lower for keyword in knowledge_keywords)
    needs_clarification = (
        len(words) < 3
        or query_lower in ambiguous_phrases
        or not should_retrieve
    ) and state.get("clarification_attempts", 0) < 2

    print(
        f"[DECISION] Query: '{query}' | Retrieve: {should_retrieve} | Clarify: {needs_clarification}"
    )

    return {
        "query": query,
        "should_retrieve": should_retrieve,
        "needs_clarification": needs_clarification,
        "preferences": state["preferences"],
    }
def ask_clarifying_question(state: AgentState) -> dict:
    """Ask the user for clarification before retrieving."""
    query = state["messages"][-1].content

    clarification_prompt = (
        "I want to make sure I point you to the right events. "
        "Could you share more details such as the city, event type, date range, or budget?"
    )

    if "city" in query.lower():
        clarification_prompt = (
            "Thanks! Could you specify the type of event (e.g., tech meetup, concert, workshop) "
            "or any budget preference?"
        )
    elif "concert" in query.lower() or "music" in query.lower():
        clarification_prompt = (
            "Are you looking for a specific city or date range for concerts?"
        )

    print("[CLARIFY] Asking user for more specific details before retrieval.")

    clarification_message = AIMessage(
        content=clarification_prompt,
        additional_kwargs={"clarification": True},
    )

    return {
        "messages": [clarification_message],
        "clarification_attempts": state.get("clarification_attempts", 0) + 1,
        "needs_clarification": False,
    }



def retrieve_documents(state: AgentState) -> dict:
    """Retrieve relevant documents from vector store."""
    query = state["query"]
    
    try:
        docs = retriever.invoke(query)
        
        print(f"[RETRIEVAL] Found {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"  - Doc {i+1}: {doc.page_content[:100]}...")
        
        return {"documents": docs}
    
    except Exception as e:
        print(f"[RETRIEVAL ERROR] {e}")
        return {"documents": []}


def generate_response(state: AgentState) -> dict:
    """Generate response using LLM with or without context."""
    messages = state["messages"]
    
    preferences = state.get("preferences") or {}
    profile_context = format_preferences(preferences)

    if state.get("documents") and len(state["documents"]) > 0:
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state["documents"])
        ])
        system_prompt = f"""You are a helpful AI assistant for event information. Use the following event details to answer the user's question accurately and concisely.

Context:
{context}

User preferences (if provided):
{profile_context or 'None provided'}

Instructions:
- Answer based on the provided event information
- Include specific details like dates, venues, prices, and cities when relevant
- If the context doesn't contain relevant information, acknowledge that
- Be concise and clear
- Format prices in Indian Rupees (₹) when mentioned"""
        
        augmented_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        print("[GENERATION] Using RAG mode with event context")
    else:
        system_prompt = f"""You are a helpful AI assistant. Answer the user's question concisely and accurately.

User preferences (if provided):
{profile_context or 'None provided'}
"""
        augmented_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        print("[GENERATION] Using chat mode without retrieval")
    
    try:
        response = llm.invoke(augmented_messages)
    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return ask_clarifying_question(state)

    if looks_like_error_page(response.content):
        print("[GENERATION] Detected HTML/error output. Falling back to clarification prompt.")
        return ask_clarifying_question(state)

    return {"messages": [response]}


# ==================== ROUTING LOGIC ====================

def route_after_decision(state: AgentState) -> str:
    """Route to clarification, retrieval or direct generation."""
    if state.get("needs_clarification"):
        return "clarify"
    return "retrieve" if state["should_retrieve"] else "generate"


# ==================== BUILD LANGGRAPH ====================

workflow = StateGraph(AgentState)

workflow.add_node("preferences", maybe_update_preferences)
workflow.add_node("decide", decide_to_retrieve)
workflow.add_node("clarify", ask_clarifying_question)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("preferences")

workflow.add_edge("preferences", "decide")

workflow.add_conditional_edges(
    "decide",
    route_after_decision,
    {
        "clarify": "clarify",
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("clarify", END)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


# ==================== HELPER FUNCTIONS ====================

def run_agent(
    user_query: str,
    history: Optional[Sequence[BaseMessage]] = None,
) -> dict:
    """Run the agent with a user query and optional prior conversation history."""
    existing_messages: List[BaseMessage] = list(history) if history else []
    clarification_attempts = sum(
        1
        for msg in existing_messages
        if isinstance(msg, AIMessage) and msg.additional_kwargs.get("clarification")
    )

    initial_state = {
        "messages": [*existing_messages, HumanMessage(content=user_query)],
        "documents": [],
        "query": "",
        "should_retrieve": False,
        "needs_clarification": False,
        "clarification_attempts": clarification_attempts,
        "preferences": load_preferences(),
    }
    
    print(f"\n{'='*70}")
    print(f"USER QUERY: {user_query}")
    print(f"{'='*70}")
    
    result = app.invoke(initial_state)
    
    print(f"\nRESULT:")
    print(f"  - Retrieved documents: {len(result.get('documents', []))}")
    print(f"  - Used retrieval: {result.get('should_retrieve', False)}")
    print(f"\nASSISTANT RESPONSE:")
    print(f"{result['messages'][-1].content}")
    print(f"{'='*70}\n")
    
    return result


# ==================== MAIN ====================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("EVENT FINDER - AGENT-BASED RAG SYSTEM WITH LANGGRAPH")
    print("Using: Sentence Transformers (embeddings) + Groq (LLM)")
    print("="*70)
    
    # Test queries
    test_queries = [
        "What technology events are happening in Mumbai?",
        "Show me music concerts",
        "What events are free to attend?",
        "Tell me about the AI and Machine Learning Workshop",
    ]
    
    print("\nRunning test queries...\n")
    
    conversation_history: List[BaseMessage] = []
    
    for query in test_queries:
        result = run_agent(query, history=conversation_history)
        conversation_history = list(result.get("messages", conversation_history))
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*70 + "\n")
    
    while True:
        user_input = input("Your query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = run_agent(user_input, history=conversation_history)
        conversation_history = list(result.get("messages", conversation_history))


if __name__ == "__main__":
    main()

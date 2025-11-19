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
from langchain_core.documents import Document
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
    is_event_query: bool


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


@tool("user_profile_tool")
def UserProfileTool(
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


@tool("event_search_tool")
def EventSearchTool(query: str) -> List[Dict[str, Any]]:
    """Search for events in the database using semantic search. Returns a list of relevant event documents."""
    try:
        docs = retriever.invoke(query)
        
        print(f"[RETRIEVAL] Found {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"  - Doc {i+1}: {doc.page_content[:100]}...")
        
        # Convert documents to a serializable format
        result = []
        for doc in docs:
            result.append({
                "page_content": doc.page_content,
                "metadata": getattr(doc, "metadata", {}) or {}
            })
        
        return result
    
    except Exception as e:
        print(f"[RETRIEVAL ERROR] {e}")
        return []


PREFERENCE_SYSTEM_PROMPT = (
    "You review the most recent user message and determine whether it contains new "
    "information about their preferred city, budget, or interests.\n"
    "- If it does, call the `user_profile_tool` tool exactly once, passing the "
    "raw user text and the current preferences.\n"
    "- If there is no new information, respond with 'No preference update needed.'"
)

preference_tools = [UserProfileTool]
preference_tool_map = {tool.name: tool for tool in preference_tools}
preference_llm = llm.bind_tools(preference_tools)

SEARCH_SYSTEM_PROMPT = (
    "You review the user's query and determine if they are asking about specific events, "
    "event details, or need information from the event database.\n"
    "- If the query asks about events (e.g., 'what events', 'show me concerts', 'find workshops'), "
    "call the `event_search_tool` with the user's query as the search term.\n"
    "- If the query is too vague or unclear, respond with 'No search needed - query too vague.'\n"
    "- If the query is a general question that doesn't require event search, respond with 'No search needed.'"
)

search_tools = [EventSearchTool]
search_tool_map = {tool.name: tool for tool in search_tools}
search_llm = llm.bind_tools(search_tools)


SEARCH_INTENT_PROMPT = """
Examples:
User: "my city is pune"
Classification: EVENT_QUERY

User: "budget under 1000"
Classification: EVENT_QUERY

User: "I want to find events"
Classification: NEEDS_DETAILS

User: "events"
Classification: NEEDS_DETAILS

User: "hi"
Classification: SMALL_TALK

User: "thanks"  
Classification: SMALL_TALK

Now classify this message. Respond with exactly one token:
- 'EVENT_QUERY' if the user is clearly asking about events or providing enough info to search.
- 'NEEDS_DETAILS' if the user mentions events but the request is too vague and needs clarification.
- 'SMALL_TALK' if the user is greeting, acknowledging, or casually reacting.

First, analyze what information the user has provided about their event search.
Then classify the intent.

Analysis:
- Does the message contain location, budget, event type, or dates? 
- Is it part of an ongoing conversation?

Classification (respond with exactly one): EVENT_QUERY, NEEDS_DETAILS, or SMALL_TALK
"""



def looks_like_error_page(content: str) -> bool:
    """Detect whether the LLM output appears to be an HTML error page."""
    if not content:
        return False

    probe = content.strip().lower()
    html_indicators = ("<!doctype", "<html", "<head", "<body")
    if any(probe.startswith(tag) for tag in html_indicators):
        return True
    return "cloudflare" in probe or "<script" in probe


EVENT_BOUNDARY_PROMPT = (
    "Classify whether the latest user message is still within scope for an events "
    "chatbot that is already in conversation with the user. Respond with exactly "
    "one token:\n"
    "- 'IN_SCOPE' if the user is greeting, acknowledging the assistant (e.g., 'hel''cool', "
    "'thanks'), making small talk, asking about events, or sharing context like city, "
    "budget, interests, or dates to guide event suggestions. Treat follow-up "
    "reactions to prior event recommendations as IN_SCOPE even if they do not mention "
    "events explicitly.\n"
    "- 'OUT_OF_SCOPE' only when the user asks for information unrelated to events or "
    "requests tasks far outside event assistance (e.g., explain quantum physics, "
    "stock advice, coding help, trivia about planets)."
)


def enforce_event_boundary(state: AgentState) -> dict:
    """Ensure the query is event-related before proceeding."""
    latest_user_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None,
    )
    if latest_user_message is None:
        return {"is_event_query": True}

    user_text = (latest_user_message.content or "").strip()
    if not user_text:
        return {"is_event_query": True}

    event_related = True
    try:
        decision = llm.invoke(
            [
                SystemMessage(content=EVENT_BOUNDARY_PROMPT),
                HumanMessage(content=user_text),
            ]
        )
        verdict = (decision.content or "").strip().lower()
        event_related = verdict.startswith("in_scope")
    except Exception as exc:
        print(f"[BOUNDARY] Boundary LLM failed: {exc}. Allowing query.")
        event_related = True

    if event_related:
        return {"is_event_query": True}

    print(f"[BOUNDARY] Blocked non-event query: '{user_text}'")
    off_topic_message = AIMessage(
        content="I'm your event chatbot. Please ask me questions related to events.",
        additional_kwargs={"off_topic": True},
    )
    return {
        "messages": [off_topic_message],
        "is_event_query": False,
        "documents": [],
        "should_retrieve": False,
        "needs_clarification": False,
    }


def classify_search_intent(text: str) -> str:
    """Use the LLM to decide how to treat the latest user query."""
    normalized = (text or "").strip()
    if not normalized:
        return "SMALL_TALK"

    try:
        decision = llm.invoke(
            [
                SystemMessage(content=SEARCH_INTENT_PROMPT),
                HumanMessage(content=normalized),
            ]
        )
        label = (decision.content or "").strip().upper()
        if label in {"EVENT_QUERY", "NEEDS_DETAILS", "SMALL_TALK"}:
            return label
    except Exception as exc:
        print(f"[SEARCH] Intent classifier failed: {exc}. Defaulting to EVENT_QUERY.")
    return "EVENT_QUERY"


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


def maybe_search_events(state: AgentState) -> dict:
    """Let the LLM decide whether to call the event search tool."""
    latest_user_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None,
    )
    if latest_user_message is None:
        return {"query": "", "should_retrieve": False}

    query = latest_user_message.content
    intent = classify_search_intent(query)
    if intent == "SMALL_TALK":
        print(f"[SEARCH] Small talk detected: '{query}' - skipping retrieval.")
        return {
            "query": query,
            "should_retrieve": False,
            "needs_clarification": False,
        }

    if intent == "NEEDS_DETAILS" and state.get("clarification_attempts", 0) < 2:
        print(f"[SEARCH] Query needs clarification: '{query}'")
        return {
            "query": query,
            "should_retrieve": False,
            "needs_clarification": True,
        }

    try:
        decision = search_llm.invoke(
            [
                SystemMessage(content=SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
    except Exception as exc:
        print(f"[SEARCH] Search LLM failed: {exc}")
        return {"query": query, "should_retrieve": False}

    tool_calls = getattr(decision, "tool_calls", None)
    if not tool_calls:
        print(f"[SEARCH] No search needed for query: '{query}'")
        return {"query": query, "should_retrieve": False}

    retrieved_docs = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        selected_tool = search_tool_map.get(tool_name)
        if selected_tool is None:
            continue

        tool_args = dict(tool_call.get("args") or {})
        search_query = tool_args.get("query", query)

        try:
            result = selected_tool.invoke({"query": search_query})
            if isinstance(result, list):
                # Convert back to document-like objects for compatibility
                retrieved_docs = [
                    Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
                    for item in result
                ]
        except Exception as exc:
            print(f"[SEARCH] Tool '{tool_name}' failed: {exc}")
            continue

    if retrieved_docs:
        print(f"[SEARCH] Retrieved {len(retrieved_docs)} documents via tool call")
        return {
            "query": query,
            "documents": retrieved_docs,
            "should_retrieve": True,
        }

    return {"query": query, "should_retrieve": False}
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

def route_after_search(state: AgentState) -> str:
    """Route to clarification or direct generation after search."""
    if state.get("needs_clarification"):
        return "clarify"
    return "generate"


def route_guardrail(state: AgentState) -> str:
    """Determine whether to continue after the guardrail."""
    return "search" if state.get("is_event_query", True) else "stop"


# ==================== BUILD LANGGRAPH ====================

workflow = StateGraph(AgentState)

workflow.add_node("preferences", maybe_update_preferences)
workflow.add_node("guardrail", enforce_event_boundary)
workflow.add_node("search", maybe_search_events)
workflow.add_node("clarify", ask_clarifying_question)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("preferences")

workflow.add_edge("preferences", "guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_guardrail,
    {
        "search": "search",
        "stop": END,
    },
)

workflow.add_conditional_edges(
    "search",
    route_after_search,
    {
        "clarify": "clarify",
        "generate": "generate"
    }
)

workflow.add_edge("clarify", END)
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
        "is_event_query": True,
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

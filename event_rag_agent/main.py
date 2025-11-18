"""Main application for Event RAG Agent using LangGraph."""

import os
from typing import TypedDict, Annotated, Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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


# ==================== AGENT STATE ====================

class AgentState(TypedDict):
    """State schema for the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list
    query: str
    should_retrieve: bool


# ==================== AGENT NODES ====================

def decide_to_retrieve(state: AgentState) -> dict:
    """Analyze query to decide if retrieval is needed."""
    query = state["messages"][-1].content
    
    knowledge_keywords = [
        "what", "how", "explain", "describe", "tell me about",
        "define", "who is", "why", "when", "where", "which",
        "event", "show", "find", "search", "list", "price"
    ]
    
    query_lower = query.lower()
    should_retrieve = any(keyword in query_lower for keyword in knowledge_keywords)
    
    print(f"[DECISION] Query: '{query}' | Retrieve: {should_retrieve}")
    
    return {
        "query": query,
        "should_retrieve": should_retrieve
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
    
    if state.get("documents") and len(state["documents"]) > 0:
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state["documents"])
        ])
        
        system_prompt = f"""You are a helpful AI assistant for event information. Use the following event details to answer the user's question accurately and concisely.

Context:
{context}

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
        system_prompt = "You are a helpful AI assistant. Answer the user's question concisely and accurately."
        
        augmented_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        print("[GENERATION] Using chat mode without retrieval")
    
    try:
        response = llm.invoke(augmented_messages)
        return {"messages": [response]}
    
    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        error_message = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
        return {"messages": [error_message]}


# ==================== ROUTING LOGIC ====================

def route_after_decision(state: AgentState) -> str:
    """Route to retrieval or direct generation."""
    return "retrieve" if state["should_retrieve"] else "generate"


# ==================== BUILD LANGGRAPH ====================

workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_to_retrieve)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("decide")

workflow.add_conditional_edges(
    "decide",
    route_after_decision,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


# ==================== HELPER FUNCTIONS ====================

def run_agent(user_query: str) -> dict:
    """Run the agent with a user query."""
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "documents": [],
        "query": "",
        "should_retrieve": False
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
    
    for query in test_queries:
        run_agent(query)
    
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
        
        run_agent(user_input)


if __name__ == "__main__":
    main()

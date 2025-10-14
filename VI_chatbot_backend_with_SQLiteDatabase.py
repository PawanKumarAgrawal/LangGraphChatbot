# ==========================================
# LangGraph Chatbot with SQLite Checkpointer
# ==========================================

# pip install langgraph-checkpoint-sqlite

# ----------- Imports -----------
# Core LangGraph / LangChain imports
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

# Typing support for defining state
from typing import TypedDict, Annotated

# Database
import sqlite3

# Load environment variables from .env file
load_dotenv()



# ----------- LLM Setup -----------
# Initialize the ChatOpenAI model
llm =  ChatOpenAI(model="gpt-4o-mini")



# ----------- State Definition -----------
# This defines the "memory state" that flows through the graph.
# It stores a list of messages (user + AI).
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# ----------- Node Function -----------
# Each node in LangGraph is a function.
# This node takes user messages, calls the LLM, and returns AI response.
def chat_node(state: ChatState):
    # Get conversation history from state
    messages = state['messages']

    # Send to LLM and get response
    response = llm.invoke(messages)

    # Return response (will be added to state)
    return {"messages": [response]}



# ----------- Database Setup -----------
# SQLite connection for storing checkpoints (chat history / threads)
conn = sqlite3.connect(database='lg_chatbot.db', check_same_thread=False)



# Ensure labels table exists (thread_id → label)
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_labels (
    thread_id TEXT PRIMARY KEY,
    label TEXT
)
""")
conn.commit()



# Create checkpointer to persist conversation states
checkpointer = SqliteSaver(conn=conn)



# ----------- Graph Setup -----------
# Create a StateGraph that represents the chatbot flow
graph = StateGraph(ChatState)

# Add nodes (functions)
graph.add_node("chat_node", chat_node)

# Add edges (flow between nodes)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile the graph into a runnable chatbot
# Pass the SQLite checkpointer so conversation history is saved
chatbot = graph.compile(checkpointer=checkpointer)



# ----------- Utility Function -----------

# Retrieve all conversation thread IDs from the SQLite checkpointer
def retrieve_all_threads():
    all_threads = set()

    # Iterate through all checkpoints and collect thread_ids
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)



def save_label(thread_id, label):
    """Save or update label for a thread"""
    conn.execute(
        "INSERT OR REPLACE INTO thread_labels (thread_id, label) VALUES (?, ?)",
        (str(thread_id), label)
    )
    conn.commit()



def get_labels():
    """Return dict of all thread_id → label"""
    rows = conn.execute("SELECT thread_id, label FROM thread_labels").fetchall()
    return {row[0]: row[1] for row in rows}














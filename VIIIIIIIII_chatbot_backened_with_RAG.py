# ===============================
# 1️⃣ Standard Library Imports
# ===============================
import os
import tempfile
import asyncio
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict


# ===============================
# 2️⃣ Environment & Config
# ===============================
from dotenv import load_dotenv


# ===============================
# 3️⃣ LangChain - LLM & Embeddings
# ===============================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool 
from langchain_core.runnables import RunnableConfig


# ===============================
# 4️⃣ LangChain - Document Processing
# ===============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===============================
# 5️⃣ LangChain - Vector Store
# ===============================
from langchain_community.vectorstores import FAISS


# ===============================
# 6️⃣ LangChain - Tools
# ===============================
from langchain_community.tools import DuckDuckGoSearchRun


# ===============================
# 7️⃣ LangGraph
# ===============================
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ===============================
# 8️⃣ MCP (Model Context Protocol)
# ===============================
from langchain_mcp_adapters.client import MultiServerMCPClient


# ===============================
# 9️⃣ Networking
# ===============================
import httpx


# ===============================
# 🔟 Database
# ===============================
import aiosqlite




# ==========================================================
# 1️⃣ Environment Setup
# ==========================================================

# Load environment variables from .env file
load_dotenv()



# ==========================================================
# 2️⃣ Dedicated Async Background Loop
# ==========================================================

# Create a dedicated async event loop running in a background thread
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True
)
_ASYNC_THREAD.start()


def run_async(coro):
    """Run a coroutine synchronously from a non-async context."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()



# ==========================================================
# 3️⃣ LLM & Embeddings Initialization
# ==========================================================

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



# ==========================================================
# 4️⃣ Thread-Based PDF Retriever Storage
# ==========================================================

# Stores retrievers per thread
_THREAD_RETRIEVERS: Dict[str, Any] = {}

# Stores metadata per thread
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None



def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass



# ==========================================================
# 5️⃣ Tool Definitions
# ==========================================================

# ---- Web Search Tool ----
search_tool = DuckDuckGoSearchRun(region="us-en")



@tool
async def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=GLOBAL_QUOTE&symbol={symbol}&apikey=MQD52N3SVW9L9SA9"
        )
        response = await client.get(url)

    return response.json()



@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF
    for this chat thread.
    """
    # Automatically extract thread_id from the graph config
    thread_id = config.get("configurable", {}).get("thread_id")

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)

    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }



# ==========================================================
# 6️⃣ MCP Client Setup
# ==========================================================

client = MultiServerMCPClient(
    {
        "maths": {
            "transport": "stdio",
            "command": "python",
            "args": ["E:/My AI YT Code/05_LangGraphChatbot/Calculator_MCP_Server.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://expense-tracker-latest-v1.fastmcp.app/mcp"
        }
    }
)



async def async_load_mcp_tools():
    """Load MCP tools asynchronously."""
    return await client.get_tools()



def load_mcp_tools():
    """Synchronous wrapper for async MCP loader."""
    return run_async(async_load_mcp_tools())



# Load MCP tools (blocking)
mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools, rag_tool]

llm_with_tools = llm.bind_tools(tools)



# ==========================================================
# 7️⃣ LangGraph State Definition
# ==========================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# ==========================================================
# 8️⃣ Graph Nodes
# ==========================================================

async def chat_node(state: ChatState):
    """LLM node that may answer directly or call a tool."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}



tool_node = ToolNode(tools)



# ==========================================================
# 9️⃣ Graph Setup & Compilation
# ==========================================================

async def setup_graph():

    conn = await aiosqlite.connect("lg_chatbot.db")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_labels (
            thread_id TEXT PRIMARY KEY,
            label TEXT
        )
    """)

    await conn.commit()

    checkpointer = AsyncSqliteSaver(conn)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile(checkpointer=checkpointer)

    return chatbot, conn, checkpointer



# ==========================================================
# 🔟 Utility Functions
# ==========================================================

async def retrieve_all_threads(checkpointer):
    """Retrieve all saved thread IDs."""
    all_threads = set()

    async for checkpoint in checkpointer.alist(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)



async def save_label(conn, thread_id, label):
    """Save or update a thread label."""
    await conn.execute(
        "INSERT OR REPLACE INTO thread_labels (thread_id, label) VALUES (?, ?)",
        (str(thread_id), label),
    )

    await conn.commit()



async def get_labels(conn):
    """Fetch all stored thread labels."""
    cursor = await conn.execute(
        "SELECT thread_id, label FROM thread_labels"
    )

    rows = await cursor.fetchall()

    return {row[0]: row[1] for row in rows}
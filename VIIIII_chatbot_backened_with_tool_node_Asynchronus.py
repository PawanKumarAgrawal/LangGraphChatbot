

# ==========================================
# LangGraph Chatbot with SQLite Checkpointer + Tools
# ==========================================

# pip install langgraph-checkpoint-sqlite

# ----------- Imports -----------
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import httpx

# Typing support for defining state
from typing import TypedDict, Annotated

# Database
import aiosqlite

# Load environment variables from .env file
load_dotenv()


# ----------- LLM Setup -----------
llm = ChatOpenAI(model="gpt-4o-mini")



# ----------- Tools Setup -----------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
async def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
async def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey= MQD52N3SVW9L9SA9"
        r = await client.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)



# ----------- State Definition -----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ----------- Node Functions -----------
async def chat_node(state: ChatState):
    """LLM node that may answer directly or call a tool"""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)



async def setup_graph():
    conn = await aiosqlite.connect("lg_chatbot.db")

    # Ensure labels table exists
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






# ----------- Utility Functions -----------
async def retrieve_all_threads(checkpointer):
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)



async def save_label(conn, thread_id, label):
    await conn.execute(
        "INSERT OR REPLACE INTO thread_labels (thread_id, label) VALUES (?, ?)",
        (str(thread_id), label)
    )
    await conn.commit()



async def get_labels(conn):
    cursor = await conn.execute("SELECT thread_id, label FROM thread_labels")
    rows = await cursor.fetchall()
    return {row[0]: row[1] for row in rows}
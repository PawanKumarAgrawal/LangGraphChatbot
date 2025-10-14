

# ==========================================
# LangGraph Chatbot with SQLite Checkpointer + Tools
# ==========================================

# pip install langgraph-checkpoint-sqlite

# ----------- Imports -----------
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Typing support for defining state
from typing import TypedDict, Annotated

# Database
import sqlite3
import requests

# Load environment variables from .env file
load_dotenv()


# ----------- LLM Setup -----------
llm = ChatOpenAI(model="gpt-4o-mini")



# ----------- Tools Setup -----------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
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
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey= MQD52N3SVW9L9SA9"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)



# ----------- State Definition -----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ----------- Node Functions -----------
def chat_node(state: ChatState):
    """LLM node that may answer directly or call a tool"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)



# ----------- Database Setup -----------
conn = sqlite3.connect(database='lg_chatbot.db', check_same_thread=False)

# Ensure labels table exists
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_labels (
    thread_id TEXT PRIMARY KEY,
    label TEXT
)
""")
conn.commit()

# Create checkpointer
checkpointer = SqliteSaver(conn=conn)



# ----------- Graph Setup -----------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)



# ----------- Utility Functions -----------
def retrieve_all_threads():
    all_threads = set()
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

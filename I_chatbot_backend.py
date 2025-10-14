# 🌟 Tutorial: Simple Chatbot with Persistence using LangGraph and LangChain 🌟

# 1️⃣ Import required libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver



# 2️⃣ Define the state structure
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# 3️⃣ Initialize the language model (LLM)
llm = ChatOpenAI(model="gpt-4o-mini")



# 4️⃣ Define the chatbot node
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}



# 5️⃣ Create the StateGraph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)



# 6️⃣ Compile the graph into a chatbot with persistence
checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)





# # 7️⃣ Visualize the Graph
# Image(chatbot.get_graph().draw_mermaid_png())



# # 8️⃣ Interactive loop with persistence
# thread_id = '1'
# config = {"configurable": {"thread_id": thread_id}}



# while True:
#     user_message = input('Type here: ')
#     print('User:', user_message)
    
#     if user_message.strip().lower() in ['leave', 'stop', 'goodbye']:
#         print('Chat ended.')
#         break

#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print('AI:', response['messages'][-1].content)

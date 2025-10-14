# streamlit run II_streamlit_frontend.py

import streamlit as st
from I_chatbot_backend import chatbot   # Import the compiled chatbot from your backend
from langchain_core.messages import HumanMessage



st.title("🤖 Welcome to My LangGraph Chatbot")

# Configuration for LangGraph persistence
CONFIG = {"configurable": {"thread_id": "thread-1"}}



# Initialize session state to store conversation history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous conversation messages (persistent across reruns)
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Input box for user
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save and display user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)



    # Get AI response from chatbot
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response["messages"][-1].content

    # Save and display AI response
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.markdown(ai_message)

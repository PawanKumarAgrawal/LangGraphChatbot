# streamlit run III_streamlit_frontend_streaming.py

import streamlit as st
from I_chatbot_backend import chatbot   # Import your LangGraph chatbot backend
from langchain_core.messages import HumanMessage

# -----------------------------------------------------------------------------
# Chatbot APP TITLE
# -----------------------------------------------------------------------------
st.title("🤖 Welcome to My LangGraph Chatbot")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {"configurable": {"thread_id": "thread-1"}}

# -----------------------------------------------------------------------------
# SESSION STATE SETUP
# -----------------------------------------------------------------------------
# Initialize message history if not already present
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# -----------------------------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# -----------------------------------------------------------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# USER INPUT
# -----------------------------------------------------------------------------
user_input = st.chat_input("Type your message here...")



# -----------------------------------------------------------------------------
# HANDLE USER INPUT
# -----------------------------------------------------------------------------
if user_input:
    # Step 1: Save & display user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)





    # Step 2: Stream assistant response
    with st.chat_message("assistant"):

        # Start streaming the response from chatbot
        response_stream = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},  #  user input
            config=CONFIG,                                     #  thread/session info
            stream_mode="messages",                            # stream tokens as messages
        )



        # Collect and render the streaming tokens one by one
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in response_stream
        )



    # Step 3: Save assistant response
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

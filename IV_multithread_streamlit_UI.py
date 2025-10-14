# streamlit run IV_multithread_streamlit_UI.py

# ==========================================
# Basic Multi-Thread Chatbot -1
# ==========================================

import streamlit as st
from I_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ------------------- Utility Functions -------------------
# These helper functions manage thread creation, reset, and
# loading previous conversation data from backend.


# Generate a unique thread id for each conversation
def generate_thread_id():
    return uuid.uuid4()

# Add a new thread id into the session list if not already present
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)



# Reset the current chat by creating a new thread and clearing history
def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.session_state['message_history'] = []



# Load saved conversation messages from backend for a given thread
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])



# ------------------- Session Setup -------------------
# Initialize session_state variables (only once per app run).
# Ensures message history, thread id, and thread list exist in memory.


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# Make sure the active thread id is registered in chat_threads
add_thread(st.session_state['thread_id'])



# ------------------- Sidebar UI -------------------
# Left panel of the app:
# - Start a new chat
# - List all existing threads (conversations)
# - Allow switching between old conversations


st.sidebar.title("Welcome to my LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Display all threads in reverse order (latest first)
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id), key=f"btn_{thread_id}"):
        # When a thread is clicked, make it active and load messages
        st.session_state['thread_id'] = thread_id
        msgs = load_conversation(thread_id)

        # Convert backend message objects into dicts for UI rendering
        history = []
        for msg in msgs:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            history.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = history



# ------------------- Main UI -------------------
# Central chat area:
# - Display previous messages (chat history)
# - Provide chat input box
# - Stream assistant responses and save them


# Render all past chat messages
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])



# Input box at the bottom for user text
user_input = st.chat_input("Type here")

if user_input:
    # Add user message to local history and display it
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)



    # Thread id passed to backend for context
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream assistant response token by token
    with st.chat_message("assistant"):
        def ai_only_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_reply = st.write_stream(ai_only_stream())

    # Save assistant's final reply to local history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})
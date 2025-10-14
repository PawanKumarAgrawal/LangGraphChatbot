# streamlit run V_renamable_multithread_streamlit_UI.py



# ==========================================
# Lecture 2: Final Chatbot with Rename + Move-to-Top
# ==========================================

import streamlit as st
from I_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ------------------- Utility Functions -------------------

def generate_thread_id():
    """Generate a new unique chat thread ID"""
    return uuid.uuid4()


def add_thread(thread_id):
    """Add a new thread ID to chat_threads list if not already present"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def move_thread_to_top(thread_id):   # NEW in Lecture 2
    """Move a given thread to the top of the list (last element shown first in reversed loop)"""
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    """Reset chat → creates new thread, clears history, moves it to top"""
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    move_thread_to_top(new_id)   # ensure new thread is at the top
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    """Load existing conversation from backend for a given thread"""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# ------------------- Rename Callback -------------------

def finish_rename(thread_id, key):   # NEW in Lecture 2
    """
    Rename chat when user presses Enter:
    - Update label in thread_labels
    - Move renamed chat to top
    - Close the edit box (editing_label = None)
    """
    new_value = st.session_state[key]
    st.session_state['thread_labels'][thread_id] = new_value
    move_thread_to_top(thread_id)
    st.session_state['editing_label'] = None



# ------------------- Session Setup -------------------

# This section makes sure that Streamlit session_state has all
# the variables we need to track chats, history, and labels.
# These are initialized only on first run.

# 1. Store the actual chat messages (user + assistant) for active thread
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# 2. Each chat belongs to a unique thread → store current active thread ID
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# 3. Keep a list of all thread IDs created so far (multiple chats possible)
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# 4. Mapping of thread_id → user-friendly label
#    Example: {"uuid-1234": "My First Question"}
#    (Lecture 2 feature: makes UI look like ChatGPT with names instead of raw IDs)
if 'thread_labels' not in st.session_state:   # NEW in Lecture 2
    st.session_state['thread_labels'] = {}

# 5. Track which thread is currently being renamed
#    If editing_label = some_thread_id → show text input box
#    If editing_label = None → no rename active
if 'editing_label' not in st.session_state:   # NEW in Lecture 2
    st.session_state['editing_label'] = None

# 6. Finally, ensure the current thread is present in chat_threads
#    and push it to the top (so latest conversation is always shown first)
add_thread(st.session_state['thread_id'])
move_thread_to_top(st.session_state['thread_id'])



# ------------------- Sidebar -------------------

st.sidebar.title("Welcome to my LangGraph Chatbot")

# Create a brand new chat
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Loop through chat threads in reverse order (last = top)
for thread_id in st.session_state['chat_threads'][::-1]:
    # Get label if available, otherwise fallback to truncated ID
    current_label = st.session_state['thread_labels'].get(
        thread_id, f"Chat {str(thread_id)[:8]}"
    )

    # Split row into 2 columns: [label button] [rename button]
    cols = st.sidebar.columns([4, 1])

    # ---- Open chat button ----
    if cols[0].button(current_label, key=f"open_{thread_id}"):
        # Set this thread as active
        st.session_state['thread_id'] = thread_id
        move_thread_to_top(thread_id)   # ensure opened chat comes on top

        # Load conversation history for this thread
        msgs = load_conversation(thread_id)
        history = []
        for m in msgs:
            role = 'user' if isinstance(m, HumanMessage) else 'assistant'
            history.append({'role': role, 'content': m.content})
        st.session_state['message_history'] = history


    # ---- Rename chat button ✏️ ----   # NEW in Lecture 2
    if cols[1].button("✏️", key=f"edit_{thread_id}"):
        st.session_state['editing_label'] = thread_id

    # ---- If rename mode is active, show input box ----   # NEW in Lecture 2
    if st.session_state['editing_label'] == thread_id:
        st.sidebar.text_input(
            "Rename Chat",
            value=current_label,
            key=f"rename_{thread_id}",
            on_change=finish_rename,
            args=(thread_id, f"rename_{thread_id}")  # pass ID + key
        )



# ------------------- Main Chat Area -------------------

# Display chat history
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

# Input box for user
user_input = st.chat_input("Type here")

if user_input:
    # Save user message in history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Config to send to backend (contains thread ID)
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream assistant response
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

    # Save assistant reply in history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})


    # ---- Auto-label chat from first user message ----   # NEW in Lecture 2
    if st.session_state['thread_id'] not in st.session_state['thread_labels']:
        st.session_state['thread_labels'][st.session_state['thread_id']] = user_input[:30] + "..."

    # Ensure this thread comes on top after message
    move_thread_to_top(st.session_state['thread_id'])

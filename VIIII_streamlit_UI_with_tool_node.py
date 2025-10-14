# streamlit run VIIII_streamlit_UI_with_tool_node.py

# ==========================================
# Tool Streaming Integration
# ==========================================

import streamlit as st
from VIII_chatbot_backened_with_tool_node import chatbot, retrieve_all_threads, save_label, get_labels
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ------------------- Utility Functions -------------------

def generate_thread_id():
    return uuid.uuid4()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def move_thread_to_top(thread_id):
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    move_thread_to_top(new_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def finish_rename(thread_id, key):
    new_value = st.session_state[key]
    st.session_state['thread_labels'][thread_id] = new_value
    save_label(thread_id, new_value)
    move_thread_to_top(thread_id)
    st.session_state['editing_label'] = None


# ------------------- Session Setup -------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_labels' not in st.session_state:
    st.session_state['thread_labels'] = get_labels()

if 'editing_label' not in st.session_state:
    st.session_state['editing_label'] = None

add_thread(st.session_state['thread_id'])
move_thread_to_top(st.session_state['thread_id'])


# ------------------- Sidebar -------------------

st.sidebar.title("Welcome to my Agentic AI Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")



for thread_id in st.session_state['chat_threads'][::-1]:
    current_label = st.session_state['thread_labels'].get(
        thread_id, f"Chat {str(thread_id)[:8]}"
    )
    
    
    cols = st.sidebar.columns([4, 1])



    if cols[0].button(current_label, key=f"open_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        move_thread_to_top(thread_id)
        msgs = load_conversation(thread_id)
        history = []
        for msg in msgs:
            if isinstance(msg, ToolMessage):
                continue  # 🚫 Skip tool messages on reload
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            history.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = history
        st.rerun()



    if cols[1].button("✏️", key=f"edit_{thread_id}"):
        st.session_state['editing_label'] = thread_id

    if st.session_state['editing_label'] == thread_id:
        st.sidebar.text_input(
            "Rename Chat",
            value=current_label,
            key=f"rename_{thread_id}",
            on_change=finish_rename,
            args=(thread_id, f"rename_{thread_id}")
        )


# ------------------- Main Chat Area -------------------

for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn_trace",
    }

    # ------------------- Assistant Streaming with Tool Integration -------------------
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # 🔧 Show tool activity
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                    st.markdown(f"**Tool `{tool_name}` executed.**")

                # 🧠 Stream assistant reply
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_reply = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution completed", state="complete", expanded=False
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})



    if st.session_state['thread_id'] not in st.session_state['thread_labels']:
        label = user_input[:30] + "..."
        st.session_state['thread_labels'][st.session_state['thread_id']] = label
        save_label(st.session_state['thread_id'], label)
        move_thread_to_top(st.session_state['thread_id'])

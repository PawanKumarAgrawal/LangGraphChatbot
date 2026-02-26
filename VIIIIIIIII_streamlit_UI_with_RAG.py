
# streamlit run VIIIIIIIII_streamlit_UI_with_RAG.py
# ==========================================

import streamlit as st
from VIIIIIIIII_chatbot_backened_with_RAG import setup_graph, retrieve_all_threads, save_label, get_labels, ingest_pdf
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import time

# ------------------- Session Setup -------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())
if 'chat_threads' not in st.session_state:
    _, conn, checkpointer = asyncio.run(setup_graph())
    st.session_state['chat_threads'] = asyncio.run(retrieve_all_threads(checkpointer))
    asyncio.run(conn.close())
if 'thread_labels' not in st.session_state:
    _, conn, checkpointer = asyncio.run(setup_graph())
    st.session_state['thread_labels'] = asyncio.run(get_labels(conn))
    asyncio.run(conn.close())
if 'editing_label' not in st.session_state:
    st.session_state['editing_label'] = None
if 'thread_docs' not in st.session_state:
    st.session_state['thread_docs'] = {}
if 'last_used' not in st.session_state:
    st.session_state['last_used'] = {}

# Helper variables
thread_key = st.session_state['thread_id']
thread_docs = st.session_state['thread_docs']

# ------------------- Utility Functions -------------------
def generate_thread_id():
    return str(uuid.uuid4())

def move_thread_to_top(thread_id):
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.session_state['chat_threads'].append(thread_id)
    st.session_state['last_used'][thread_id] = time.time()

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    if new_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(new_id)
    move_thread_to_top(new_id)
    st.session_state['message_history'] = []

async def load_conversation_async(thread_id):
    chatbot, conn, checkpointer = await setup_graph()
    state = await chatbot.aget_state(config={'configurable': {'thread_id': thread_id}})
    await conn.close()
    return state.values.get('messages', [])

def finish_rename(thread_id, key):
    new_value = st.session_state[key].strip()
    if not new_value:
        new_value = f"Chat {str(thread_id)[:8]}"
    st.session_state['thread_labels'][thread_id] = new_value
    _, conn, _ = asyncio.run(setup_graph())
    asyncio.run(save_label(conn, thread_id, new_value))
    asyncio.run(conn.close())
    move_thread_to_top(thread_id)
    st.session_state['editing_label'] = None

# ------------------- Sidebar -------------------
st.sidebar.title("LangGraph Chatbot\n RAG + MCP Server + Tools")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# PDF status & upload
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# ────────────────────────────────────────────────
# Only renamable thread labels – no UUIDs visible
# ────────────────────────────────────────────────
st.sidebar.header("My Conversations")

# Sort by last used (most recent first)
sorted_threads = sorted(
    st.session_state['chat_threads'],
    key=lambda tid: st.session_state['last_used'].get(tid, 0),
    reverse=True
)

if not sorted_threads:
    st.sidebar.info("No conversations yet. Start chatting!")
else:
    for thread_id in sorted_threads:
        # Always have a label
        current_label = st.session_state['thread_labels'].get(
            thread_id,
            f"Chat {str(thread_id)[:8]}"
        )

        cols = st.sidebar.columns([5, 1])

        # Open chat button
        if cols[0].button(
            current_label,
            key=f"open_{thread_id}",
            use_container_width=True
        ):
            st.session_state['thread_id'] = thread_id
            move_thread_to_top(thread_id)
            msgs = asyncio.run(load_conversation_async(thread_id))
            history = []
            for msg in msgs:
                if isinstance(msg, ToolMessage):
                    continue
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                history.append({'role': role, 'content': msg.content})
            st.session_state['message_history'] = history
            st.rerun()

        # Rename button
        if cols[1].button("✏️", key=f"edit_{thread_id}", help="Rename"):
            st.session_state['editing_label'] = thread_id

        # Rename input field appears inline
        if st.session_state.get('editing_label') == thread_id:
            st.sidebar.text_input(
                "Rename",
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

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        async def ai_only_stream():
            chatbot, conn, checkpointer = await setup_graph()
            try:
                async for chunk, metadata in chatbot.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
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
                    if isinstance(chunk, AIMessage):
                        yield chunk.content
            finally:
                await conn.close()

        collected_content = []
        placeholder = st.empty()

        async def collect_and_display():
            async for chunk in ai_only_stream():
                collected_content.append(chunk)
                placeholder.markdown("".join(collected_content))

        asyncio.run(collect_and_display())

        ai_reply = "".join(collected_content)
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution completed", state="complete", expanded=False
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})

    # Auto-label new threads with first user message
    if st.session_state['thread_id'] not in st.session_state['thread_labels']:
        label = (user_input.strip()[:35] + "...") if len(user_input) > 35 else user_input.strip()
        if not label:
            label = "New conversation"
        st.session_state['thread_labels'][st.session_state['thread_id']] = label
        _, conn, _ = asyncio.run(setup_graph())
        asyncio.run(save_label(conn, st.session_state['thread_id'], label))
        asyncio.run(conn.close())
        move_thread_to_top(st.session_state['thread_id'])
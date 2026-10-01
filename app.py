import streamlit as st
from document_loader_New_RAGAS_Evaluation import init_rag_system, rag_pipeline
import os
import uuid
import json
from datetime import datetime

st.set_page_config(
    page_title="Student Anxiety and Wellbeing Chatbot",
    page_icon="😊",
    layout="wide"
)

CHAT_DIR = "chat_histories"
os.makedirs(CHAT_DIR, exist_ok=True)



@st.cache_resource
def load_backend():
    with st.spinner("Loading AI model and vector DB"):
        retriever, llm, safety_guard = init_rag_system()
        return retriever, llm, safety_guard

retriever, llm, safety_guard = load_backend()

def get_chats():
    files = [f for f in os.listdir(CHAT_DIR) if f.endswith(".json")]
    chats = []
    for file in files:
        with open(os.path.join(CHAT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            chats.append({
                "id": data.get("id"),
                "title": data.get("title","New Chat"),
                "timestamp": data.get("timestamp",0)
            })
    return sorted(chats, key=lambda x:x["timestamp"], reverse=True)

def load_chat(chat_id):
    filepath = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get("messages",[])
    return []

def save_chat(chat_id, messages, title):
    filepath = os.path.join(CHAT_DIR, f"{chat_id}.json")
    data = {
        "id": chat_id,
        "title": title,
        "timestamp": datetime.now().timestamp(),
        "messages": messages
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image("WIL CHATBOT LOGO.png", use_container_width=True)
    st.title("Chat History")

    if st.button(" New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()

    all_chats = get_chats()
    for chat in all_chats:
        raw_title = str(chat.get('title','New Chat'))
        title_snippet = f"{raw_title[:25]}" + ("..." if len(raw_title) > 25 else "")


        if st.button(title_snippet, key=chat["id"], use_container_width=True):
            st.session_state.current_chat_id = chat["id"]
            st.session_state.messages = load_chat(chat["id"])
            st.rerun()

st.title("Student Anxiety and Wellbeing Chatbot")
st.caption("An AI assistant to help you find accurate services information.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_pipeline(prompt, retriever, llm, safety_guard)

            st.markdown(response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    chat_title = st.session_state.messages[0] if st.session_state.messages else "New Chat"

    if st.session_state.messages:
        first_msg = st.session_state.messages[0]

        chat_title = first_msg.get("content", "New Chat")
    else:
        chat_title = "New Chat"
    
    save_chat(st.session_state.current_chat_id, st.session_state.messages, chat_title)

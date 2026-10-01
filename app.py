"""Streamlit interface for the student wellbeing support-navigation chatbot."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag_pipeline import init_rag_system, run_navigation

BASE_DIR = Path(__file__).resolve().parent
LOGO_FILE = BASE_DIR / 'assets' / 'wil_chatbot_logo.png'
CHAT_HISTORY_DIR = BASE_DIR / 'chat_histories'
SAVE_CHATS = os.getenv('WIL_SAVE_CHATS', '0').strip() == '1'

APP_TITLE = 'Tertiary Student Wellbeing and Support Navigation Chatbot'

st.set_page_config(
    page_title=APP_TITLE,
    page_icon='🧭',
    layout='wide',
)

@st.cache_resource(show_spinner='Loading the support-navigation chatbot...')
def load_backend():
    """Load the RAG pipeline once for the current Streamlit process."""
    return init_rag_system()

def list_saved_chats():
    """Load locally saved chats when chat persistence is enabled."""
    if not SAVE_CHATS:
        return []

    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    chats = []

    for file_path in CHAT_HISTORY_DIR.glob('*.json'):
        try:
            chat = json.loads(file_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue

        if chat.get('id') == file_path.stem:
            chats.append(chat)

    return sorted(
        chats,
        key=lambda chat: chat.get('timestamp', ''),
        reverse=True,
    )

def save_current_chat():
    """Save the current chat locally only when explicitly enabled."""
    if not SAVE_CHATS:
        return

    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    first_question = next(
        (
            message['content']
            for message in st.session_state.messages
            if message['role'] == 'user'
        ),
        'New Chat',
    )

    chat = {
        'id': st.session_state.chat_id,
        'title': first_question[:80],
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'messages': st.session_state.messages,
    }

    output_file = CHAT_HISTORY_DIR / f"{st.session_state.chat_id}.json"
    output_file.write_text(
        json.dumps(chat, indent=2),
        encoding='utf-8',
    )

def archive_current_chat():
    """Keep the current chat available within the active Streamlit session."""
    if not st.session_state.get('messages'):
        return

    if not st.session_state.get('chat_id'):
        return

    first_question = next(
        (
            message['content']
            for message in st.session_state.messages
            if message['role'] == 'user'
        ),
        'New Chat',
    )

    st.session_state.chat_archive[st.session_state.chat_id] = {
        'id': st.session_state.chat_id,
        'title': first_question[:80],
        'messages': list(st.session_state.messages),
    }

def start_new_chat():
    """Archive the visible conversation and start an empty chat."""
    if 'chat_archive' not in st.session_state:
        st.session_state.chat_archive = {}

    archive_current_chat()

    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

def open_chat(chat):
    """Open a previous chat in the current Streamlit session."""
    archive_current_chat()

    st.session_state.chat_id = chat['id']
    st.session_state.messages = chat.get('messages', [])

if 'chat_archive' not in st.session_state:
    st.session_state.chat_archive = {}

if 'chat_id' not in st.session_state:
    start_new_chat()

with st.sidebar:
    if LOGO_FILE.is_file():
        st.image(str(LOGO_FILE), width='stretch')

    st.markdown('### Chat history')
    st.caption(
        'This is a local research prototype. Use fictional scenarios rather than '
        'real private student information.'
    )

    if SAVE_CHATS:
        st.info('Local chat saving is enabled.')
    else:
        st.info('Disk-based chat saving is off. Messages stay in this session only.')

    if st.button('New Chat', width='stretch'):
        start_new_chat()
        st.rerun()

    if SAVE_CHATS:
        recent_chats = list_saved_chats()
    else:
        recent_chats = list(st.session_state.chat_archive.values())[::-1]

    for chat in recent_chats:
        if st.button(
            chat.get('title', 'New Chat')[:29],
            key=chat['id'],
            width='stretch',
        ):
            open_chat(chat)
            st.rerun()

st.title(APP_TITLE)
st.caption(
    'Research prototype for navigating published student support pathways. '
    'It is not an AI therapist, diagnostic service or emergency service.'
)
st.warning(
    'If someone is in immediate danger in Australia, call 000. '
    'For 24/7 crisis support, contact Lifeline on 13 11 14.'
)

try:
    retriever, llm, safety_guard = load_backend()
except Exception as error:
    st.error(
        'The local model or support-service index could not be loaded. '
        'Check the setup instructions in README.md.'
    )
    st.code(str(error))
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask about an official student support pathway...')

if prompt:
    st.session_state.messages.append(
        {
            'role': 'user',
            'content': prompt,
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Checking the available support information...'):
            result = run_navigation(
                prompt,
                retriever,
                llm,
                safety_guard,
            )

        st.markdown(result['answer'])

        if result['route'] == 'answered':
            st.caption(
                'The published eligibility, contact information and official source '
                'are shown above. Check the live source before acting.'
            )

    st.session_state.messages.append(
        {
            'role': 'assistant',
            'content': result['answer'],
        }
    )

    save_current_chat()

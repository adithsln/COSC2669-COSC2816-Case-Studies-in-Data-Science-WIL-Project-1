"""Streamlit interface for the student wellbeing support-navigation chatbot."""

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag_pipeline import init_rag_system, run_navigation
from support_navigation import canonical_institution_reply

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / 'assets'
LOGO_FILE = ASSETS_DIR / 'wil_chatbot_logo.png'
BROWSER_LOGO_FILE = ASSETS_DIR / 'chatbot_logo_browser.png'
DISPLAY_LOGO_FILE = (
    BROWSER_LOGO_FILE if BROWSER_LOGO_FILE.is_file() else LOGO_FILE
)
CHAT_HISTORY_DIR = BASE_DIR / 'chat_histories'
SAVE_CHATS = os.getenv('WIL_SAVE_CHATS', '0').strip() == '1'

APP_TITLE = 'Tertiary Student Wellbeing and Support Navigation Chatbot'
MAIN_LOGO_WIDTH = 240

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(LOGO_FILE) if LOGO_FILE.is_file() else '🧭',
    layout='wide',
)

st.markdown(
    """
    <style>
    .emergency-banner {
        background-color: #107D78;
        color: white;
        padding: 0.85rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner='Loading the support-navigation chatbot...')
def load_backend():
    """Load the RAG pipeline once for the current Streamlit process."""
    return init_rag_system()

def image_data_uri(file_path):
    """Return a local PNG as a data URI for centred HTML rendering."""
    encoded = base64.b64encode(file_path.read_bytes()).decode('ascii')
    return f'data:image/png;base64,{encoded}'

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
    st.session_state.pending_question = None
    st.session_state.pending_institution = None

def open_chat(chat):
    """Open a previous chat in the current Streamlit session."""
    archive_current_chat()

    st.session_state.chat_id = chat['id']
    st.session_state.messages = chat.get('messages', [])
    st.session_state.pending_question = None
    st.session_state.pending_institution = None

if 'chat_archive' not in st.session_state:
    st.session_state.chat_archive = {}

if 'chat_id' not in st.session_state:
    start_new_chat()

if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None

if 'pending_institution' not in st.session_state:
    st.session_state.pending_institution = None

with st.sidebar:
    if LOGO_FILE.is_file():
        st.image(str(LOGO_FILE), width='stretch')

    st.markdown('### Chat history')
    st.caption(
        'This is a local research prototype. Use fictional scenarios only. '
        'Do not enter real private student information.'
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

if DISPLAY_LOGO_FILE.is_file():
    st.markdown(
        f'''
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <img
                src="{image_data_uri(DISPLAY_LOGO_FILE)}"
                style="width: {MAIN_LOGO_WIDTH}px; max-width: 45vw; height: auto;"
                alt="Tertiary Student Wellbeing and Support Navigation Chatbot logo"
            >
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.markdown(
    f'''
    <h1 style="
        text-align: center;
        margin-top: 0.25rem;
        margin-bottom: 0.65rem;
    ">
        {APP_TITLE}
    </h1>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    '''
    <p style="
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 1100px;
        margin: 0 auto 1.25rem auto;
    ">
        Research prototype for navigating published student support pathways.
        It is not an AI therapist, diagnostic service or emergency service.
        Current coverage includes RMIT University, RMIT UP, Monash University
        and the University of Melbourne.
    </p>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="emergency-banner" role="alert">
        If someone is in immediate danger in Australia, call <strong>000</strong>.
        For 24/7 crisis support, contact Lifeline on <strong>13 11 14</strong>.
    </div>
    """,
    unsafe_allow_html=True,
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

    navigation_prompt = prompt
    institution_reply = canonical_institution_reply(prompt)
    pending_question = st.session_state.get('pending_question')
    pending_institution = st.session_state.get('pending_institution')

    if pending_question and institution_reply:
        navigation_prompt = (
            f"{pending_question} "
            f"My university is {institution_reply}."
        )
        st.session_state.pending_question = None
        st.session_state.pending_institution = None

    elif pending_question:
        st.session_state.pending_question = None

    elif pending_institution and not institution_reply:
        navigation_prompt = (
            f"My university is {pending_institution}. "
            f"{prompt}"
        )
        st.session_state.pending_institution = None

    elif pending_institution and institution_reply:
        st.session_state.pending_institution = institution_reply

    with st.chat_message('assistant'):
        with st.spinner('Checking the available support information...'):
            result = run_navigation(
                navigation_prompt,
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

    if (
        result['route'] == 'clarify'
        and result['answer'].startswith('Which university do you attend:')
    ):
        st.session_state.pending_question = navigation_prompt
        st.session_state.pending_institution = None

    elif (
        result['route'] == 'clarify'
        and result['answer'].startswith(
            'What kind of wellbeing or student support are you looking for?'
        )
        and institution_reply
    ):
        st.session_state.pending_institution = institution_reply
        st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            'role': 'assistant',
            'content': result['answer'],
        }
    )

    save_current_chat()

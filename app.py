"""Streamlit demonstration UI for the source-grounded student support navigation prototype.

Use fictional test inputs. Not deployed as a production counselling or emergency tool.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from rag_pipeline import init_rag_system, run_navigation
from service_catalog import load_catalog

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / 'assets' / 'WIL CHATBOT LOGO.png'
CHAT_DIR = ROOT / 'chat_histories'
# Off by default: avoid silently retaining sensitive student wellbeing questions.
PERSIST_CHATS = os.getenv('WIL_SAVE_CHATS', '0').strip() == '1'

st.set_page_config(page_title='Tertiary Student Wellbeing and Support Navigation Chatbot',
                   page_icon='🧭', layout='wide')


@st.cache_resource(show_spinner='Loading the local support-navigation model and reviewed source snapshot...')
def load_backend():
    return init_rag_system()


def list_chats():
    if not PERSIST_CHATS:
        return []
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    chats = []
    for path in CHAT_DIR.glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('id') == path.stem:
                chats.append(data)
        except (ValueError, OSError):
            continue
    return sorted(chats, key=lambda x: x.get('timestamp', ''), reverse=True)


def save_chat():
    if not PERSIST_CHATS:
        return
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    chat_id = st.session_state.chat_id
    first = next((m['content'] for m in st.session_state.messages if m['role'] == 'user'), 'New Chat')
    data = {'id': chat_id, 'title': first[:80],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'messages': st.session_state.messages}
    (CHAT_DIR / (chat_id + '.json')).write_text(json.dumps(data, indent=2), encoding='utf-8')


def new_chat():
    if 'chat_archive' not in st.session_state:
        st.session_state.chat_archive = {}
    if st.session_state.get('messages') and st.session_state.get('chat_id'):
        old_id = st.session_state.chat_id
        first = next((m['content'] for m in st.session_state.messages if m['role']=='user'), 'New Chat')
        st.session_state.chat_archive[old_id] = {'id': old_id, 'title': first[:80], 'messages': list(st.session_state.messages)}
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []


if 'chat_id' not in st.session_state:
    new_chat()

with st.sidebar:
    if LOGO.is_file():
        st.image(str(LOGO), width='stretch')
    st.markdown('### Chat history')
    st.caption('This is a local demonstration. Please enter fictional scenarios, not real private student data.')
    if PERSIST_CHATS:
        st.info('Local JSON chat saving has been enabled by the operator.')
    else:
        st.info('Disk-based chat saving is OFF. Messages are held in this session only.')
    if st.button('New Chat', width='stretch'):
        new_chat()
        st.rerun()
    recent_chats = list_chats() if PERSIST_CHATS else list(st.session_state.get('chat_archive', {}).values())[::-1]
    for chat in recent_chats:
        if st.button(chat.get('title', 'New Chat')[:29], key=chat['id'], width='stretch'):
            # Archive the currently visible session before switching.
            if st.session_state.get('messages'):
                current_first = next((m['content'] for m in st.session_state.messages if m['role']=='user'), 'New Chat')
                st.session_state.chat_archive[st.session_state.chat_id] = {
                    'id': st.session_state.chat_id, 'title': current_first[:80], 'messages': list(st.session_state.messages)}
            st.session_state.chat_id = chat['id']
            st.session_state.messages = chat.get('messages', [])
            st.rerun()

st.title('Tertiary Student Wellbeing and Support Navigation Chatbot')
st.caption('Research prototype: helps find published support pathways. Not an AI therapist, diagnosis service or emergency service.')
st.warning('If someone is in immediate danger in Australia, call 000. For 24/7 crisis support, Lifeline: 13 11 14.')

try:
    retriever, llm, safety_guard = load_backend()
except Exception as exc:
    st.error('The local model or official-source index could not be loaded. Check README.md and run the setup commands.')
    st.code(str(exc))
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask about an official university support pathway...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Checking the available source snapshot...'):
            result = run_navigation(prompt, retriever, llm, safety_guard)
        st.markdown(result['answer'])
        if result['route'] == 'answered':
            st.caption('Official source and published eligibility shown above. Check the live official page before acting.')
    st.session_state.messages.append({'role': 'assistant', 'content': result['answer']})
    save_chat()

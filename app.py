import streamlit as st
import uuid
from main import main_workflow
from langchain_mongodb import MongoDBChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
db_name = "chat_memory"
collection_name = "messages"

st.title("🧠 Researcher-X: AI Research Agent")

st.subheader("🔑 Session Management")
session_id_input = st.text_input("Enter Session ID (or leave empty for a new one):", key="session_input")

if session_id_input and session_id_input != st.session_state.get("session_id"):
    st.session_state.session_id = session_id_input.strip()
    st.session_state.chat_history = []
elif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.write(f"**Current Session ID:** `{st.session_state.session_id}`")

message_history = MongoDBChatMessageHistory(
    connection_string=MONGO_CONNECTION_STRING,
    session_id=st.session_state.session_id,
    database_name=db_name,
    collection_name=collection_name
)

st.session_state.chat_history = message_history.messages

for msg in st.session_state.chat_history:
    if msg.type == "human":
        with st.chat_message("user"):
            st.write(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg.content, unsafe_allow_html=True)

query = st.chat_input("Type your message...")

if query:
    with st.status("🔄 Generating response...", state="running"):
        result = main_workflow(st.session_state.session_id, query, message_history)

    message_history.add_user_message(query)
    message_history.add_ai_message(result)

    st.session_state.chat_history = message_history.messages  

    st.rerun()
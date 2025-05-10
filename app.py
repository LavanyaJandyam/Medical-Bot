
import streamlit as st
from orchestrate import ask_with_rag

st.set_page_config(page_title="📚 Gemini RAG Chat", layout="wide")
st.title("💬 Chat with Gemini + ChromaDB")

# Session state for message history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user's question
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    try:
        with st.spinner("Thinking..."):
            response = ask_with_rag(user_input, top_k=3)
        st.chat_message("ai").markdown(response)
        st.session_state.chat_history.append(("ai", response))
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display previous messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("ai").markdown(message)
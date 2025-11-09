import os

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_float import float_init, float_parent, float_css_helper


load_dotenv()
float_init(theme=True, include_unstable_primary=False)
st.set_page_config(page_title="Docs Agent", layout="wide")

BACKEND_URL = os.environ["API_URL"]

# init session state
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Upload and chat with your documents")
tab1, tab2 = st.tabs(["📄 Upload & Process", "💬 Chat"])

# ========== TAB 1: UPLOAD ==========
with tab1:
    st.header("Upload a document")
    uploaded = st.file_uploader("Upload a file to add to the vector store", type=["txt", "ppt", "pptx", "md", "pdf"])

    if uploaded is not None:
        if st.button("Send to backend"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
            except Exception as e:
                st.error(f"Error contacting backend: {e}")
            else:
                if resp.status_code == 202:
                    data = resp.json()
                    st.session_state.task_id = data["task_id"]
                    st.info("File accepted ✅ Processing on server...")
                else:
                    st.error(f"Upload failed: {resp.text}")

    # show processing status if we have a task_id
    if st.session_state.task_id:
        status_resp = requests.get(f"{BACKEND_URL}/status/{st.session_state.task_id}").json()
        status = status_resp.get("status")

        if status == "processing":
            st.info("⏳ Still processing your file...")
        elif status == "success":
            st.success("✅ File processed and added to vector DB!")
        elif status == "failed":
            st.error(f"⚠️ Error while processing: {status_resp.get('error')}")
        else:
            st.warning("Task not found.")


# ========== TAB 2: CHAT ==========
with tab2:
    # Get user input
    with st.container():
        query = st.chat_input("Ask something")
        chat_bar_css = float_css_helper(width="40rem", bottom="0.5rem", left="30rem", transition=0)
        float_parent(css=chat_bar_css)

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})

    # Render chat history
    with st.container(height=380):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Get response from agent
    if query:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": query},
                timeout=60
            )
            resp.raise_for_status()
            answer = resp.json().get("answer", "No answer returned")
        except Exception as e:
            answer = f"Error contacting backend: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import uuid
import streamlit as st

from config import APP_NAME
from graph.builder import build_graph
from utils.langsmith_tracing import traceable, maybe_tracing_context


st.set_page_config(
    page_title=APP_NAME,
    page_icon="💬",
    layout="wide",
)


graph = build_graph()
upload_dir = Path("artifacts/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


@traceable(
    name="streamlit_graph_invoke",
    tags=["app", "streamlit", "support-agent"],
    metadata={"surface": "streamlit-chat"},
)
def invoke_graph_with_trace(graph_obj, state: dict, config: dict):
    return graph_obj.invoke(state, config=config)


def remove_image():
    st.session_state.uploaded_image_path = None
    st.session_state.uploaded_image_name = None
    st.session_state.uploader_key += 1
    st.rerun()


def reset_chat():
    st.session_state.messages = []
    st.session_state.uploaded_image_path = None
    st.session_state.uploaded_image_name = None
    st.session_state.uploader_key += 1
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


with st.sidebar:
    st.title("Support Agent")
    st.caption("Upload an optional image and manage the conversation.")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        save_path = upload_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_image_path = str(save_path)
        st.session_state.uploaded_image_name = uploaded_file.name

    if st.session_state.uploaded_image_path:
        st.image(
            st.session_state.uploaded_image_path,
            caption="Current image",
            use_container_width=True,
        )
        st.caption(f"Attached: {st.session_state.uploaded_image_name}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Remove image", use_container_width=True, on_click=remove_image)
    with col2:
        st.button("Reset chat", use_container_width=True, on_click=reset_chat)

    with st.expander("Session info"):
        st.caption(f"user_id: {st.session_state.user_id}")
        st.caption(f"session_id: {st.session_state.session_id}")
        st.caption(f"thread_id: {st.session_state.thread_id}")


st.title(APP_NAME)
st.caption("Chat naturally, or ask order, refund, return, damage, or policy questions.")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi, I’m Support Intelligence Agent. You can chat normally, ask support questions, or upload an image from the sidebar."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Type your message...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = {
                "query": user_query,
                "image_path": st.session_state.uploaded_image_path,
                "chat_history": st.session_state.messages[-8:],
                "messages": st.session_state.messages[-8:],
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "thread_id": st.session_state.thread_id,
            }

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                }
            }

            with maybe_tracing_context(project_name="support-intelligence-agent"):
                result = invoke_graph_with_trace(graph, state, config)

            answer = result.get("final_answer", "I could not generate a response.")
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

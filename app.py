import streamlit as st

from agent import run_agent


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="centered"
)


# -------------------------
# Title
# -------------------------

st.title("🤖 AI Research Agent")

st.write(
    "Ask questions and let the agent decide when to use "
    "web search or the weather tool."
)


# -------------------------
# Chat History
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# Chat Input
# -------------------------

user_query = st.chat_input(
    "Ask something..."
)


# -------------------------
# Process Query
# -------------------------

if user_query:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Agent is thinking..."):

            try:
                answer = run_agent(user_query)

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                error_message = f"Error: {str(e)}"

                st.error(error_message)
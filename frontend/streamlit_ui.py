import streamlit as st
import requests
import uuid

# Assign a unique user ID for session tracking
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Streamlit App Title
st.title("Financio - AI Financial Assistant")

# Sidebar for model parameters
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("Enter your query"):
    # Add user message to session
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the payload to send to the backend
    payload = {
        "user_id": st.session_state["user_id"],
        "question": prompt,
        "chat_history": [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state["messages"]
        ],  # Ensuring the correct format
    }

    try:
        # Query FastAPI backend
        response = requests.post("http://localhost:8001/query", json=payload)
        response.raise_for_status()  # Raise error if request fails
        data = response.json()
        bot_response = data.get("response", "I'm sorry, I couldn't fetch a response.")

    except requests.exceptions.RequestException as e:
        bot_response = f"Error communicating with the backend: {e}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Update chat history
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

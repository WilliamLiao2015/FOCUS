import streamlit as st
import requests

from datetime import datetime


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    st.title("FOCUS: Find Out Characters Under Specification")
    if "log" not in st.session_state: st.session_state.log = []

    for message in st.session_state.log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.log.append({
            "type": "query",
            "role": "user",
            "content": user_input,
            "time": get_time()
        })
        response = requests.post("http://localhost:8080/v1/chat/completions", json={"log": st.session_state.log}, headers={"Content-Type": "application/json"})
        response = response.json()

        with st.chat_message("assistant"):
            st.markdown(response["log"][-1]["content"])

        st.session_state.log = response["log"]

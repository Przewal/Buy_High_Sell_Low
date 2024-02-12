from openai import OpenAI
import streamlit as st
import pandas as pd

st.title("Midas Bot")

client = OpenAI(api_key="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")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "you are a financial analyst that recieves metrics about stocks and aims to help the user understand the performance in a concise manner. Here are the portfolio stats and stocks:"}]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
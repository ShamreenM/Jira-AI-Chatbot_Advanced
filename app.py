import streamlit as st
from JiraAIChatbot import retrieve_docs
from dotenv import load_dotenv
import os

env_path = "\key.env"
load_dotenv(dotenv_path=env_path)
apiKey = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="JIRA AI Chatbot")

st.title("🤖 JIRA AI Chatbot")

user_input = st.text_input("Ask your JIRA question:")

if st.button("Submit"):
    if user_input:
        with st.spinner("🔍 Searching JIRA issues..."):
            result = retrieve_docs(user_input, apiKey)

        st.markdown("### Response")
        st.text(result)

    else:
        st.warning("Please enter a question.")
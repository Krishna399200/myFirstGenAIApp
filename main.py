import os
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- API KEY ---
# Prefer Streamlit secrets (local .streamlit/secrets.toml) then fall back to env var
OPENAI_API_KEY = None
try:
    # st.secrets is available in Streamlit runtime and when running locally with .streamlit/secrets.toml
    if getattr(st, "secrets", None) and st.secrets.get("OPENAI_API_KEY"):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
except Exception:
    # safe fallback if st.secrets is not available or not configured
    OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    # try environment variable fallback
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set. Add it to .streamlit/secrets.toml or set OPENAI_API_KEY env var.")
else:
    # make sure it's exported to process env for any libraries that expect it
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Prompt Template ---
tweet_template = """
Generate {number} engaging tweets on the topic: {topic}.
Return each tweet on a new line.
"""
tweet_prompt = PromptTemplate.from_template(tweet_template)

# --- OpenAI GPT Model ---
model = ChatOpenAI(
    model="gpt-4o-mini",   # very fast + cheap, change model if you want
    temperature=0.7
)

# --- LCEL Chain (Prompt → Model) ---
tweet_chain = tweet_prompt | model

# --- Streamlit UI ---
st.set_page_config(page_title="Tweet Generator (OpenAI)", layout="centered")
st.title("Tweet Generator App")
st.subheader("Powered by OpenAI GPT Models")

topic = st.text_input("Enter a topic for the tweets:")
number = st.slider("Number of tweets:", 1, 10, 3)

if st.button("Generate Tweets"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        response = tweet_chain.invoke({
            "number": number,
            "topic": topic
        })

        text = response.content if hasattr(response, "content") else str(response)
        tweets = [t.strip() for t in text.split("\n") if t.strip()]

        for i, tw in enumerate(tweets, start=1):
            st.markdown(f"**Tweet {i}:** {tw}")

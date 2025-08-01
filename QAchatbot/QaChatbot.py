import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_anthropic    import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os

##page Configurations

st.set_page_config(
    page_title="Langchain chatbot",
    page_icon=":robot:",
    layout="wide"
)


with st.sidebar:
    st.header("Settings")
    # Set the API key for Anthropic
    anthro_key = st.text_input("Enter your Anthropic API Key", type="password" ,help="Get your API key from https://console.anthropic.com/keys")
    if anthro_key:
        os.environ["ANTHROPIC_API_KEY"] = anthro_key

    ##model selection
    model = st.selectbox(
        "Select Model", 
        ["claude-2", "claude-instant-100k", "claude-1", "claude-1.3"],
    )

#title
st.title("Langchain Chatbot with Anthropic")
st.markdown("This is a simple chatbot using Langchain and Anthropic's Claude model.")

#model selection

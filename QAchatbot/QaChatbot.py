import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_anthropic  import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os

##page Configurations

st.set_page_config(
    page_title="Langchain chatbot",
    page_icon=":robot:",
    layout="wide"
)

#title
st.title("Langchain Chatbot with Anthropic")
st.markdown("This is a simple chatbot using Langchain and Anthropic's Claude model.")

with st.sidebar:
    st.header("Settings")
    # Set the API key for Anthropic
    anthro_key = st.text_input("Enter your Anthropic API Key", type="password" ,help="Get your API key from https://console.anthropic.com/keys")
    if anthro_key:
        os.environ["ANTHROPIC_API_KEY"] = anthro_key

    ##model selection
    model = st.selectbox(
        "Select Model", 
        ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        index=0,
    )
    #clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()



#intialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


##intialize LLM
@st.cache_resource
def get_llm(model_name,api_key):
    """Initialize the chat model with the given model name and API key."""
    if not api_key:
         return None

    ##initialize the chat model
    llm=ChatAnthropic(
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000,
            api_key=api_key,
            streaming=True
        )
    return llm



import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_anthropic  import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

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
def get_chain(model_name,api_key):
    """Initialize the chat model with the given model name and API key."""
    if not api_key:
        return None

    ##initialize the chat model
    llm=ChatAnthropic(
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000,
            api_key=api_key,
            streaming=True,
        )

    #create prompt template
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, powered by anthropic ,Answer the user's questions to the best of your ability."),
            ("user", "{question}"), 
            ("assistant", "{answer}"),
        ]        
    )

    #create chain
    chain = prompt | llm | StrOutputParser()
    return chain

#get the chain
chain = get_chain(model, anthro_key)

if not chain:
    st.warning("Please enter a valid API key to continue.")
    st.markdown("You can get your API key from [Anthropic Console](https://console.anthropic.com/keys).")
else:
    #display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.write(message["content"])

    #user input
    user_input = st.chat_input("Ask a question")

    if user_input:
       st.session_state.messages.append({"role": "user", "content": user_input})
       with st.chat_message("user"):
            st.write(user_input)
    

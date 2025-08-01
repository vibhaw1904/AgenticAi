# QA Chatbot

A simple and intuitive chatbot web application built with Streamlit and Langchain, powered by Anthropic's Claude models.

## Features

- **Multiple Claude Models**: Choose from various Claude models including Sonnet, Haiku, and Opus
- **Real-time Streaming**: Get responses as they're generated with streaming support  
- **Chat History**: Maintains conversation context throughout the session
- **User-friendly Interface**: Clean Streamlit interface with sidebar settings
- **API Key Security**: Secure API key input with password masking

## Prerequisites

- Python 3.7+
- Anthropic API key (get one from [Anthropic Console](https://console.anthropic.com/keys))

## Installation

1. Install required dependencies:
```bash
pip install streamlit langchain langchain-anthropic langchain-core
```

2. Run the application:
```bash
streamlit run QaChatbot.py
```

## Usage

1. **Set API Key**: Enter your Anthropic API key in the sidebar
2. **Select Model**: Choose your preferred Claude model from the dropdown
3. **Start Chatting**: Type your questions in the chat input
4. **Clear History**: Use the "Clear Chat" button to reset the conversation

## Available Models

- `claude-3-5-sonnet-20241022` (Default)
- `claude-3-5-haiku-20241022`  
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`

## Configuration

The chatbot uses these default settings:
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 1000 per response
- **Streaming**: Enabled for real-time responses

## Example Questions

- What is the capital of France?
- Explain the theory of relativity
- What is the largest planet in our solar system?
- How many planets are there in our solar system?
- What is the largest animal on Earth?

## Technical Details

- **Framework**: Streamlit for web interface
- **LLM Integration**: Langchain with Anthropic ChatAnthropic
- **Response Parsing**: StrOutputParser for clean text output
- **Session Management**: Streamlit session state for chat history

## Error Handling

The application includes error handling for:
- Invalid API keys
- Network connectivity issues
- Model response failures

## Security

- API keys are handled securely with password input masking
- Keys are stored only in session environment variables
- No persistent storage of sensitive information

---
*Powered by [Langchain](https://langchain.com/) and [Anthropic](https://www.anthropic.com/)*
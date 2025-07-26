# Nugen.in Custom LLM Integration with LangChain

This project demonstrates how to integrate the Nugen.in API with LangChain to create a custom Large Language Model (LLM) that can be used seamlessly with LangChain's ecosystem.

## 🎯 Overview

The `nugenin.ipynb` notebook contains a complete implementation of a custom LLM that:
- Connects to the Nugen.in API
- Implements LangChain's LLM interface
- Supports both synchronous and streaming responses
- Uses environment variables for secure credential management
- Provides comprehensive error handling

## 📋 Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Nugen.in API account and API key
- Internet connection for API calls

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd CustomLLm
```

### 2. Install Dependencies
The notebook will automatically install required packages:
- `langchain` and `langchain-core`
- `python-dotenv` for environment variables
- `requests` for API calls
- `pydantic` for data validation

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
NUGEN_API_KEY=your-actual-api-key-here
NUGEN_MODEL_NAME=nugen-flash-instruct
```

### 4. Run the Notebook
Open `nugenin.ipynb` in Jupyter and run the cells sequentially.

## 📚 Code Explanation

### Cell 1: Environment Setup
```python
# Install and import required packages for environment variables
try:
    from dotenv import load_dotenv
    print("✅ python-dotenv already installed")
except ImportError:
    # Auto-install if missing
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

import os
load_dotenv()  # Load variables from .env file
```

**Purpose:** 
- Ensures `python-dotenv` is installed for environment variable management
- Loads credentials from `.env` file securely
- Prevents hardcoding sensitive information in the notebook

### Cell 2: NugenLLM Class Definition
```python
import requests
import json
from typing import Any, Dict, Iterator, List, Optional

# LangChain imports with fallbacks for different versions
try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
except ImportError:
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.llms.base import LLM
    # Fallback for older versions
    class GenerationChunk:
        def __init__(self, text: str):
            self.text = text

class NugenLLM(LLM):
    """Custom LLM implementation for Nugen.in API."""
    
    # Pydantic field definitions for validation
    api_key: str = Field(description="API key for Nugen.in service")
    model_name: str = Field(default="default", description="Name of the model to use")
    base_url: str = Field(default="https://api.dev-nugen.in", description="Base URL for the API")
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
```

**Purpose:**
- **Import Management:** Handles different LangChain versions gracefully
- **Class Definition:** Creates a custom LLM that inherits from LangChain's base LLM class
- **Field Validation:** Uses Pydantic for automatic parameter validation
- **Compatibility:** Ensures the class works across different LangChain installations

#### Core Methods Implementation

##### `_call` Method - Synchronous API Calls
```python
def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
    """Run the LLM using Nugen.in API."""
    
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{self.base_url}/api/v3/inference/completions"
    
    payload = {
        "model": self.model_name,
        "prompt": prompt,
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
    }
```

**How it connects to your custom model:**
1. **Authentication:** Uses Bearer token authentication with your API key
2. **Endpoint:** Calls the Nugen.in completions endpoint (`/api/v3/inference/completions`)
3. **Payload:** Sends your prompt along with model configuration
4. **Response Parsing:** Extracts the generated text from the API response

##### `_stream` Method - Streaming Responses
```python
def _stream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[GenerationChunk]:
    """Stream the LLM using Nugen.in API."""
    
    payload = {
        "model": self.model_name,
        "prompt": prompt,
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "stream": True  # Enable streaming
    }
    
    # Process streaming response
    for line in response.iter_lines():
        if line.startswith('data: '):
            # Parse each chunk and yield it
            chunk = GenerationChunk(text=text)
            yield chunk
```

**Streaming Connection:**
1. **Stream Flag:** Adds `"stream": True` to enable real-time responses
2. **Line Processing:** Parses Server-Sent Events (SSE) format
3. **Chunk Yielding:** Provides tokens as they're generated
4. **Real-time Display:** Enables character-by-character output

### Cell 3: Model Initialization and Testing
```python
# Get credentials from environment variables
api_key = os.getenv("NUGEN_API_KEY")
model_name = os.getenv("NUGEN_MODEL_NAME", "nugen-flash-instruct")

if not api_key:
    print("❌ NUGEN_API_KEY not found in environment variables!")
else:
    # Create model instance
    model = NugenLLM(
        api_key=api_key,
        model_name=model_name, 
        temperature=0.7,
        max_tokens=500
    )
    
    # Test basic functionality
    response = model.invoke("What is artificial intelligence?")
```

**Connection Process:**
1. **Credential Loading:** Safely retrieves API key from environment
2. **Model Instantiation:** Creates NugenLLM instance with your specific model
3. **Validation:** Pydantic automatically validates all parameters
4. **Testing:** Makes actual API call to verify connection

### Cell 4: Advanced Testing (Streaming & LangChain Integration)
```python
# Test streaming functionality
for chunk in model.stream("Tell me a short story about AI"):
    print(chunk.text, end="", flush=True)

# Test LangChain integration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "Answer this question: {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=model, prompt=prompt)

result = chain.run(question="What is machine learning?")
```

**Integration Features:**
1. **Streaming Test:** Demonstrates real-time token generation
2. **LangChain Compatibility:** Shows integration with LangChain chains
3. **Prompt Templates:** Uses LangChain's templating system
4. **Chain Execution:** Runs complex workflows with your custom model

## 🔧 How the Connection Works

### 1. Authentication Flow
```
Your Notebook → Environment Variables → API Key → Nugen.in API
```

### 2. API Communication
```
LangChain Request → NugenLLM._call() → HTTP POST → Nugen.in → Response → LangChain
```

### 3. Data Flow
```
User Prompt → JSON Payload → API Call → Model Inference → Text Response → LangChain Output
```

## 🛠️ Configuration Options

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NUGEN_API_KEY` | Your Nugen.in API key | ✅ Yes | None |
| `NUGEN_MODEL_NAME` | Model identifier | ❌ No | `nugen-flash-instruct` |
| `NUGEN_BASE_URL` | API base URL | ❌ No | `https://api.dev-nugen.in` |

### Model Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `temperature` | Creativity/randomness | 0.7 | 0.0 - 2.0 |
| `max_tokens` | Maximum response length | 1000 | 1 - 4096 |

## 🔍 Troubleshooting

### Common Issues

1. **Validation Error: api_key field required**
   ```bash
   # Solution: Check your .env file
   cat .env
   # Ensure NUGEN_API_KEY is set
   ```

2. **API Error: 401 Unauthorized**
   ```bash
   # Solution: Verify your API key
   # Check Nugen.in dashboard for correct key
   ```

3. **ImportError: No module named 'langchain'**
   ```bash
   # Solution: Install LangChain
   pip install langchain langchain-core
   ```

4. **Streaming not working**
   - Check if your model supports streaming
   - Verify network connection stability
   - Ensure proper response parsing

## 📝 Example Usage

### Basic Usage
```python
from nugenin import NugenLLM

model = NugenLLM(api_key="your-key", model_name="nugen-flash-instruct")
response = model.invoke("Explain quantum computing")
print(response)
```

### With LangChain Chains
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=model, memory=memory)

response = conversation.predict(input="Hello, how are you?")
```

### Streaming Example
```python
for chunk in model.stream("Write a poem about technology"):
    print(chunk.text, end="", flush=True)
```

## 🔒 Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Use environment variables** - Keep credentials out of code
3. **Rotate API keys regularly** - Update in Nugen.in dashboard
4. **Limit API key permissions** - Use principle of least privilege
5. **Monitor API usage** - Check for unexpected calls

## 📚 API Documentation Reference

- **Nugen.in API Docs:** [https://docs.nugen.in/](https://docs.nugen.in/)
- **LangChain Custom LLMs:** [https://python.langchain.com/docs/how_to/custom_llm/](https://python.langchain.com/docs/how_to/custom_llm/)
- **Pydantic Documentation:** [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the Nugen.in API documentation
3. Open an issue in this repository
4. Contact Nugen.in support for API-related issues

---

**Note:** This implementation is designed to be educational and production-ready. The code includes proper error handling, validation, and follows LangChain's best practices for custom LLM implementations.
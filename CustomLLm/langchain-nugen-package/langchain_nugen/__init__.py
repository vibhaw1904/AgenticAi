"""LangChain integration for Nugen.in LLMs."""

from langchain_nugen.chat_models.nugen import ChatNugen
from langchain_nugen.llms.nugen import NugenLLM

__version__ = "0.1.0"
__all__ = ["ChatNugen", "NugenLLM"]
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import Config

class LLMFactory:
    @staticmethod
    def get_groq_llm(temperature=None):
        return ChatGroq(
            model=Config.GROQ_MODEL,
            temperature=temperature or Config.TEMPERATURE,
            groq_api_key=Config.GROQ_API_KEY
        )
    
    @staticmethod
    def get_gemini_llm(temperature=None):
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=temperature or Config.TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY
        )
    
    @staticmethod
    def get_embeddings():
        # Use HuggingFace embeddings instead of Gemini to avoid async issues
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
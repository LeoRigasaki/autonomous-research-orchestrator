import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Database
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    
    # LLM Models
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Agent Settings
    MAX_ITERATIONS = 5
    TEMPERATURE = 0.3
    
    @classmethod
    def validate_keys(cls):
        required_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
        missing = [key for key in required_keys if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required API keys: {missing}")
        return True
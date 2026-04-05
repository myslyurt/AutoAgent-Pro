"""
config.py — Application Configuration
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Models
    OPENAI_API_KEY: str = "sk-your-key-here"
    ANTHROPIC_API_KEY: str = ""
    SUPERVISOR_MODEL: str = "gpt-4o"
    TRIAGE_MODEL: str = "gpt-4o-mini"

    # Memory
    REDIS_URL: str = "redis://localhost:6379"
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # External APIs (leave empty for mock mode)
    CRM_API_URL: str = ""
    CRM_API_KEY: str = ""
    ORDER_API_URL: str = ""
    ORDER_API_KEY: str = ""
    PAYMENT_API_URL: str = ""
    PAYMENT_API_KEY: str = ""
    HELPDESK_API_URL: str = ""
    HELPDESK_API_KEY: str = ""

    # Agent Settings
    MAX_AGENT_ITERATIONS: int = 10
    ESCALATION_THRESHOLD: float = 0.7
    MOCK_APIS: bool = True  # Set False in production
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

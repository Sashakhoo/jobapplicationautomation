import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Career Intelligence Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./career_assistant.db")
    
    # Redis for caching and queues
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # AI Services - Now supports multiple providers
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    
    # DeepSeek Configuration (Free API alternative)
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    # AI Service Selection
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "openai")  # openai, deepseek, or local
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Local AI (if using Ollama, LocalAI, etc.)
    LOCAL_AI_BASE_URL: Optional[str] = os.getenv("LOCAL_AI_BASE_URL", "http://localhost:11434")
    LOCAL_AI_MODEL: str = os.getenv("LOCAL_AI_MODEL", "llama2")
    
    # Job Search APIs
    LINKEDIN_CLIENT_ID: Optional[str] = os.getenv("LINKEDIN_CLIENT_ID")
    LINKEDIN_CLIENT_SECRET: Optional[str] = os.getenv("LINKEDIN_CLIENT_SECRET")
    INDEED_PUBLISHER_ID: Optional[str] = os.getenv("INDEED_PUBLISHER_ID")
    GLASSDOOR_PARTNER_ID: Optional[str] = os.getenv("GLASSDOOR_PARTNER_ID")
    
    # Email
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: Optional[str] = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    
    # Limits
    DAILY_JOB_SEARCH_LIMIT: int = 50
    DAILY_APPLICATION_LIMIT: int = 10
    MIN_MATCH_SCORE: float = 0.75  # Only recommend jobs with 75%+ match
    
    # Search Settings
    SEARCH_RADIUS_MILES: int = 50
    JOB_FRESHNESS_DAYS: int = 30
    
    # Resume Optimization
    RESUME_OPTIMIZATION_MODES: dict = {
        "light": "Minimal changes, keyword optimization only",
        "moderate": "Reordering and rewording for better impact",
        "aggressive": "Full AI-powered restructuring"
    }
    
    def get_ai_config(self):
        """Get AI configuration based on selected provider"""
        if self.AI_PROVIDER == "deepseek":
            return {
                "api_key": self.DEEPSEEK_API_KEY,
                "model": self.DEEPSEEK_MODEL,
                "base_url": self.DEEPSEEK_BASE_URL,
                "provider": "deepseek"
            }
        elif self.AI_PROVIDER == "local":
            return {
                "api_key": None,
                "model": self.LOCAL_AI_MODEL,
                "base_url": self.LOCAL_AI_BASE_URL,
                "provider": "local"
            }
        else:  # Default to OpenAI
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "base_url": self.OPENAI_BASE_URL,
                "provider": "openai"
            }
    
    class Config:
        env_file = ".env"

settings = Settings()
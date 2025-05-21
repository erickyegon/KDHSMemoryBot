import os
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings using Pydantic for validation and type safety."""

    # API Keys
    euriai_api_key: SecretStr = Field(
        default=SecretStr(""),
        env="EURIAI_API_KEY",
        description="API key for Euriai services"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL",
        description="Model to use for embeddings"
    )
    chat_model: str = Field(
        default="gpt-4.1-nano",
        env="CHAT_MODEL",
        description="Model to use for chat completions"
    )

    # Generation Settings
    temperature: float = Field(
        default=0.7,
        env="TEMPERATURE",
        description="Temperature for text generation"
    )
    max_tokens: int = Field(
        default=800,
        env="MAX_TOKENS",
        description="Maximum tokens for text generation"
    )

    # Memory Settings
    memory_k: int = Field(
        default=5,
        env="MEMORY_K",
        description="Number of relevant memories to retrieve"
    )
    history_limit: int = Field(
        default=20,
        env="HISTORY_LIMIT",
        description="Maximum number of conversation turns to keep"
    )

    # Vector Store Settings
    vector_dim: int = Field(
        default=1536,
        env="VECTOR_DIM",
        description="Dimension of embedding vectors"
    )

    # Database Settings
    qdrant_host: str = Field(
        default="localhost",
        env="QDRANT_HOST",
        description="Qdrant server host"
    )
    qdrant_port: int = Field(
        default=6333,
        env="QDRANT_PORT",
        description="Qdrant server port"
    )
    qdrant_collection: str = Field(
        default="memories",
        env="QDRANT_COLLECTION",
        description="Qdrant collection name for memories"
    )

    # User Identity
    user_identity: str = Field(
        default="",
        env="USER_IDENTITY",
        description="User identity information"
    )

    # Application Settings
    app_name: str = Field(
        default="Memory Chatbot",
        env="APP_NAME",
        description="Name of the application"
    )

    # Logging Settings
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    log_file: str = Field(
        default="logs/app.log",
        env="LOG_FILE",
        description="Log file path"
    )

    # API Endpoints
    euriai_embed_url: str = Field(
        default="https://api.euron.one/api/v1/euri/alpha/embeddings",
        env="EURIAI_EMBED_URL",
        description="Euriai embeddings API endpoint"
    )
    euriai_completion_url: str = Field(
        default="https://api.euron.one/api/v1/completions",
        env="EURIAI_COMPLETION_URL",
        description="Euriai completions API endpoint"
    )
    euriai_chat_url: str = Field(
        default="https://api.euron.one/api/v1/chat/completions",
        env="EURIAI_CHAT_URL",
        description="Euriai chat completions API endpoint"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"
    }

# Create settings instance
settings = Settings()

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

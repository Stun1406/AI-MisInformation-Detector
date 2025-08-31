from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database settings
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # Qdrant settings
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_COLLECTION_NAME: str

    # API settings
    API_HOST: str
    API_PORT: int
    API_TITLE: str
    API_VERSION: str

    # ML settings
    EMBEDDING_MODEL: str
    CLASSIFICATION_MODEL: str
    MAX_SEQUENCE_LENGTH: int

    # App settings
    DEBUG: bool
    LOG_LEVEL: str
    BATCH_SIZE: int
    SIMILARITY_THRESHOLD: float

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

settings = Settings()
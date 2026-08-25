from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IHearYou API"
    app_version: str = "0.1.0"
    cors_origins: list[str] = ["*"]

    model_path: str = "models/bisindo_translator.pth"
    image_model_path: str = "models/pipeline_mlp.pkl"
    image_metadata_path: str = "data/model_metadata.json"

    ws_max_clients: int = 100
    sequence_length: int = 30
    min_frames_for_inference: int = 15
    prediction_stability_threshold: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

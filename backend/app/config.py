import json
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    API_KEY: str = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
    CORS_ORIGINS: str = "http://localhost:3000"
    DATA_ROOT: str = "./data"
    MAX_UPLOAD_MB: int = 600
    NNUNET_COMMAND_TEMPLATE: str = "[]"
    INFERENCE_TIMEOUT_SECONDS: int = 3600

    @property
    def cors_origins(self):
        return [x.strip() for x in self.CORS_ORIGINS.split(",") if x.strip()]

    @property
    def command_template(self):
        try:
            command = json.loads(self.NNUNET_COMMAND_TEMPLATE)
        except json.JSONDecodeError as e:
            raise ValueError("NNUNET_COMMAND_TEMPLATE must be a JSON array") from e
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ValueError("NNUNET_COMMAND_TEMPLATE must be a JSON array of strings")
        return command

settings = Settings()

"""Configuration settings for LegiScan AI Governance Bills Tracker."""
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

class ConfigManager:
    def __init__(self):
        """
        Initialize configuration with profile-specific settings.

        Args:
            profile (str): Configuration profile (production, development, testing)
        """
        self._load_base_config()

    def _load_base_config(self):
        """Load base configuration that applies to all profiles."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

    def reload(self):
        """Reload configuration."""
        self._load_base_config()
        self._load_profile_config(self.profile)
        self._validate_config()

    def __str__(self) -> str:
        """Return string representation of non-sensitive config."""
        sensitive_keys = ["OPENAI_API_KEY", "LEGISCAN_API_KEY"]
        config_str = f"Configuration Profile: {self.profile}\n"
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key in sensitive_keys:
                config_str += f"{key}: {'*' * 8}\n"
            else:
                config_str += f"{key}: {value}\n"
        return config_str

# Create default instance
config = ConfigManager()
"""Database configuration."""

import os
from pathlib import Path


class DatabaseConfig:
    """Database configuration settings."""

    # Default database path relative to OneTrainer root
    DEFAULT_DB_PATH = "workspace/onetrainer.db"

    @classmethod
    def get_db_path(cls) -> Path:
        """Get database path from environment or default."""
        env_path = os.environ.get("ONETRAINER_DB_PATH")
        if env_path:
            return Path(env_path)

        # Default to workspace directory
        onetrainer_root = Path(__file__).parent.parent.parent
        return onetrainer_root / cls.DEFAULT_DB_PATH

    @classmethod
    def get_db_url(cls) -> str:
        """Get SQLAlchemy database URL."""
        db_path = cls.get_db_path()
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    @classmethod
    def is_db_enabled(cls) -> bool:
        """Check if database storage is enabled."""
        # Can be disabled via environment variable
        return os.environ.get("ONETRAINER_DB_ENABLED", "1").lower() in ("1", "true", "yes")

"""
OneTrainer Persistence Layer

SQLite-based storage with SQLAlchemy ORM for presets, concepts, samples,
training runs, and generated samples. Includes full audit trail with
version history and rollback capability.
"""

from .database import get_engine, get_session, init_database
from .config import DatabaseConfig

__all__ = [
    'get_engine',
    'get_session',
    'init_database',
    'DatabaseConfig',
]

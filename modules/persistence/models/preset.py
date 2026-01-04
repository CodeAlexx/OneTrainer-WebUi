"""Preset model for training configurations."""

import json
from datetime import datetime
from typing import Optional, List, Any, Dict, TYPE_CHECKING

from sqlalchemy import String, Text, Boolean, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, SoftDeleteMixin

if TYPE_CHECKING:
    from .training_run import TrainingRun


class Preset(Base, TimestampMixin, SoftDeleteMixin):
    """Training configuration preset."""

    __tablename__ = 'presets'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Indexed fields for querying
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    training_method: Mapped[str] = mapped_column(String(100), nullable=False)
    base_model_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    peft_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Full config as JSON blob
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    # Metadata
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    training_runs: Mapped[List["TrainingRun"]] = relationship(
        "TrainingRun",
        back_populates="preset",
        lazy="dynamic"
    )

    __table_args__ = (
        Index('idx_presets_model_type', 'model_type'),
        Index('idx_presets_training_method', 'training_method'),
        Index('idx_presets_deleted', 'deleted_at'),
    )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get the config as a dictionary."""
        return json.loads(self.config_json)

    def set_config_dict(self, config: Dict[str, Any]) -> None:
        """Set the config from a dictionary."""
        self.config_json = json.dumps(config)

    def get_tags_list(self) -> List[str]:
        """Get tags as a list."""
        if not self.tags:
            return []
        return json.loads(self.tags)

    def set_tags_list(self, tags: List[str]) -> None:
        """Set tags from a list."""
        self.tags = json.dumps(tags)

    def to_train_config(self) -> Any:
        """Convert to TrainConfig object."""
        from modules.util.config.TrainConfig import TrainConfig
        config = TrainConfig.default_values()
        config.from_dict(self.get_config_dict())
        return config

    @classmethod
    def from_train_config(
        cls,
        config: Any,
        name: str,
        description: Optional[str] = None,
        is_builtin: bool = False,
        created_by: Optional[str] = None
    ) -> "Preset":
        """Create Preset from TrainConfig."""
        config_dict = config.to_dict()
        return cls(
            name=name,
            description=description,
            model_type=str(config.model_type.name) if hasattr(config.model_type, 'name') else str(config.model_type),
            training_method=str(config.training_method.name) if hasattr(config.training_method, 'name') else str(config.training_method),
            base_model_name=config.base_model_name,
            peft_type=str(config.peft_type.name) if hasattr(config, 'peft_type') and config.peft_type else None,
            config_json=json.dumps(config_dict),
            config_version=getattr(config, 'config_version', 10),
            is_builtin=is_builtin,
            created_by=created_by
        )

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        name: str,
        description: Optional[str] = None,
        is_builtin: bool = False,
        created_by: Optional[str] = None
    ) -> "Preset":
        """Create Preset from a config dictionary."""
        return cls(
            name=name,
            description=description,
            model_type=config_dict.get('model_type', 'UNKNOWN'),
            training_method=config_dict.get('training_method', 'UNKNOWN'),
            base_model_name=config_dict.get('base_model_name'),
            peft_type=config_dict.get('peft_type'),
            config_json=json.dumps(config_dict),
            config_version=config_dict.get('__version', 10),
            is_builtin=is_builtin,
            created_by=created_by
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'training_method': self.training_method,
            'base_model_name': self.base_model_name,
            'peft_type': self.peft_type,
            'config': self.get_config_dict(),
            'config_version': self.config_version,
            'is_builtin': self.is_builtin,
            'is_favorite': self.is_favorite,
            'tags': self.get_tags_list(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by,
        }

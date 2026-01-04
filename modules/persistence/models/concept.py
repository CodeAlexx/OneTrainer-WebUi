"""Concept model for dataset definitions."""

import json
from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy import String, Text, Boolean, Integer, Float, Index
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, SoftDeleteMixin


class Concept(Base, TimestampMixin, SoftDeleteMixin):
    """Training concept (dataset definition)."""

    __tablename__ = 'concepts'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)

    # Key indexed fields
    concept_type: Mapped[str] = mapped_column(String(50), nullable=False, default='STANDARD')
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Nested configs as JSON
    image_config_json: Mapped[str] = mapped_column(Text, nullable=False, default='{}')
    text_config_json: Mapped[str] = mapped_column(Text, nullable=False, default='{}')

    # Stats as JSON blob (computed values)
    concept_stats_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Core settings
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    include_subdirectories: Mapped[bool] = mapped_column(Boolean, default=False)
    image_variations: Mapped[int] = mapped_column(Integer, default=1)
    text_variations: Mapped[int] = mapped_column(Integer, default=1)
    balancing: Mapped[float] = mapped_column(Float, default=1.0)
    balancing_strategy: Mapped[str] = mapped_column(String(50), default='REPEATS')
    loss_weight: Mapped[float] = mapped_column(Float, default=1.0)

    # Full config JSON (for complete restoration)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    __table_args__ = (
        Index('idx_concepts_name', 'name'),
        Index('idx_concepts_path', 'path'),
        Index('idx_concepts_type', 'concept_type'),
        Index('idx_concepts_deleted', 'deleted_at'),
    )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full config as a dictionary."""
        return json.loads(self.config_json)

    def set_config_dict(self, config: Dict[str, Any]) -> None:
        """Set the full config from a dictionary."""
        self.config_json = json.dumps(config)

    def get_image_config(self) -> Dict[str, Any]:
        """Get the image config as a dictionary."""
        return json.loads(self.image_config_json)

    def set_image_config(self, config: Dict[str, Any]) -> None:
        """Set the image config from a dictionary."""
        self.image_config_json = json.dumps(config)

    def get_text_config(self) -> Dict[str, Any]:
        """Get the text config as a dictionary."""
        return json.loads(self.text_config_json)

    def set_text_config(self, config: Dict[str, Any]) -> None:
        """Set the text config from a dictionary."""
        self.text_config_json = json.dumps(config)

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get concept stats as a dictionary."""
        if not self.concept_stats_json:
            return None
        return json.loads(self.concept_stats_json)

    def set_stats(self, stats: Dict[str, Any]) -> None:
        """Set concept stats from a dictionary."""
        self.concept_stats_json = json.dumps(stats)

    def to_concept_config(self) -> Any:
        """Convert to ConceptConfig object."""
        from modules.util.config.ConceptConfig import ConceptConfig
        config = ConceptConfig.default_values()
        config.from_dict(self.get_config_dict())
        return config

    @classmethod
    def from_concept_config(cls, config: Any) -> "Concept":
        """Create Concept from ConceptConfig."""
        config_dict = config.to_dict()
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Concept":
        """Create Concept from a config dictionary."""
        image_config = config_dict.get('image', {})
        text_config = config_dict.get('text', {})
        stats = config_dict.get('concept_stats')

        return cls(
            name=config_dict.get('name', 'Unnamed'),
            path=config_dict.get('path', ''),
            concept_type=config_dict.get('type', 'STANDARD'),
            enabled=config_dict.get('enabled', True),
            image_config_json=json.dumps(image_config),
            text_config_json=json.dumps(text_config),
            concept_stats_json=json.dumps(stats) if stats else None,
            seed=config_dict.get('seed'),
            include_subdirectories=config_dict.get('include_subdirectories', False),
            image_variations=config_dict.get('image_variations', 1),
            text_variations=config_dict.get('text_variations', 1),
            balancing=config_dict.get('balancing', 1.0),
            balancing_strategy=config_dict.get('balancing_strategy', 'REPEATS'),
            loss_weight=config_dict.get('loss_weight', 1.0),
            config_json=json.dumps(config_dict),
            config_version=config_dict.get('__version', 2)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert concept to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'concept_type': self.concept_type,
            'enabled': self.enabled,
            'image_config': self.get_image_config(),
            'text_config': self.get_text_config(),
            'stats': self.get_stats(),
            'seed': self.seed,
            'include_subdirectories': self.include_subdirectories,
            'image_variations': self.image_variations,
            'text_variations': self.text_variations,
            'balancing': self.balancing,
            'balancing_strategy': self.balancing_strategy,
            'loss_weight': self.loss_weight,
            'config': self.get_config_dict(),
            'config_version': self.config_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

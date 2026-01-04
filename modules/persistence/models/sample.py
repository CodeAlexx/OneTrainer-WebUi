"""Sample model for inference/sampling configurations."""

import json
from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy import String, Text, Boolean, Integer, Float, Index
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, SoftDeleteMixin


class Sample(Base, TimestampMixin, SoftDeleteMixin):
    """Inference/sampling configuration."""

    __tablename__ = 'samples'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Key fields for querying
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    random_seed: Mapped[bool] = mapped_column(Boolean, default=False)
    diffusion_steps: Mapped[int] = mapped_column(Integer, default=20)
    cfg_scale: Mapped[float] = mapped_column(Float, default=7.0)

    # Additional sampling parameters
    noise_scheduler: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    frames: Mapped[int] = mapped_column(Integer, default=1)
    length: Mapped[float] = mapped_column(Float, default=10.0)

    # Full config as JSON
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Metadata
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index('idx_samples_enabled', 'enabled'),
        Index('idx_samples_deleted', 'deleted_at'),
    )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full config as a dictionary."""
        return json.loads(self.config_json)

    def set_config_dict(self, config: Dict[str, Any]) -> None:
        """Set the full config from a dictionary."""
        self.config_json = json.dumps(config)

    def to_sample_config(self) -> Any:
        """Convert to SampleConfig object."""
        from modules.util.config.SampleConfig import SampleConfig
        config = SampleConfig.default_values()
        config.from_dict(self.get_config_dict())
        return config

    @classmethod
    def from_sample_config(cls, config: Any, name: Optional[str] = None) -> "Sample":
        """Create Sample from SampleConfig."""
        config_dict = config.to_dict()
        return cls.from_dict(config_dict, name=name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], name: Optional[str] = None) -> "Sample":
        """Create Sample from a config dictionary."""
        return cls(
            name=name,
            prompt=config_dict.get('prompt', ''),
            negative_prompt=config_dict.get('negative_prompt'),
            width=config_dict.get('width', 512),
            height=config_dict.get('height', 512),
            seed=config_dict.get('seed', 42),
            random_seed=config_dict.get('random_seed', False),
            diffusion_steps=config_dict.get('diffusion_steps', 20),
            cfg_scale=config_dict.get('cfg_scale', 7.0),
            noise_scheduler=config_dict.get('noise_scheduler'),
            frames=config_dict.get('frames', 1),
            length=config_dict.get('length', 10.0),
            config_json=json.dumps(config_dict),
            config_version=config_dict.get('__version', 0),
            enabled=config_dict.get('enabled', True)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'random_seed': self.random_seed,
            'diffusion_steps': self.diffusion_steps,
            'cfg_scale': self.cfg_scale,
            'noise_scheduler': self.noise_scheduler,
            'frames': self.frames,
            'length': self.length,
            'config': self.get_config_dict(),
            'config_version': self.config_version,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

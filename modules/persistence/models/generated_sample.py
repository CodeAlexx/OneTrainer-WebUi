"""Generated sample model for training output artifacts."""

import json
from datetime import datetime
from typing import Optional, Any, Dict, TYPE_CHECKING

from sqlalchemy import String, Text, Integer, Float, ForeignKey, Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .training_run import TrainingRun


class GeneratedSample(Base):
    """Output artifact from sampling during training."""

    __tablename__ = 'generated_samples'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('training_runs.id', ondelete='CASCADE'),
        nullable=False
    )

    # Generation context
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    global_step: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sample parameters
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    diffusion_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cfg_scale: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Output file
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'image', 'video', 'audio'
    file_format: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 'png', 'jpg', 'mp4', etc.
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Additional metadata as JSON
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    generated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # Relationships
    training_run: Mapped["TrainingRun"] = relationship(
        "TrainingRun",
        back_populates="generated_samples"
    )

    __table_args__ = (
        Index('idx_generated_samples_run', 'training_run_id'),
        Index('idx_generated_samples_epoch', 'epoch'),
        Index('idx_generated_samples_step', 'global_step'),
    )

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata as a dictionary."""
        if not self.metadata_json:
            return None
        return json.loads(self.metadata_json)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata from a dictionary."""
        self.metadata_json = json.dumps(metadata)

    @classmethod
    def from_file(
        cls,
        training_run_id: int,
        epoch: int,
        global_step: int,
        file_path: str,
        file_name: str,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> "GeneratedSample":
        """Create a GeneratedSample from a file."""
        import os
        from pathlib import Path

        path = Path(file_path)
        file_format = path.suffix.lstrip('.').lower() if path.suffix else None

        # Determine file type from format
        file_type = None
        if file_format in ('png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'):
            file_type = 'image'
        elif file_format in ('mp4', 'webm', 'avi', 'mov'):
            file_type = 'video'
        elif file_format in ('mp3', 'wav', 'flac', 'ogg'):
            file_type = 'audio'

        # Get file size if file exists
        file_size = None
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)

        return cls(
            training_run_id=training_run_id,
            epoch=epoch,
            global_step=global_step,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            file_format=file_format,
            file_size_bytes=file_size,
            prompt=prompt,
            seed=seed,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert generated sample to dictionary for API responses."""
        return {
            'id': self.id,
            'training_run_id': self.training_run_id,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'seed': self.seed,
            'width': self.width,
            'height': self.height,
            'diffusion_steps': self.diffusion_steps,
            'cfg_scale': self.cfg_scale,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'file_format': self.file_format,
            'file_size_bytes': self.file_size_bytes,
            'metadata': self.get_metadata(),
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
        }

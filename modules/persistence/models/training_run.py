"""Training run model for session history."""

import json
from datetime import datetime
from typing import Optional, List, Any, Dict, TYPE_CHECKING

from sqlalchemy import String, Text, Boolean, Integer, Float, ForeignKey, Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .preset import Preset
    from .concept import Concept
    from .sample import Sample
    from .generated_sample import GeneratedSample


class TrainingRun(Base, TimestampMixin):
    """Training session history."""

    __tablename__ = 'training_runs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to preset
    preset_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('presets.id', ondelete='SET NULL'),
        nullable=True
    )
    preset_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Run identification
    run_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    workspace_dir: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_model_destination: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration snapshot (full config at time of run)
    config_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Status tracking
    # Status values: pending, starting, training, stopping, completed, error, cancelled
    status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')

    # Progress metrics
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Loss metrics (stored as JSON for flexibility)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_smooth_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    preset: Mapped[Optional["Preset"]] = relationship(
        "Preset",
        back_populates="training_runs"
    )
    concepts: Mapped[List["TrainingRunConcept"]] = relationship(
        "TrainingRunConcept",
        back_populates="training_run",
        cascade="all, delete-orphan"
    )
    samples: Mapped[List["TrainingRunSample"]] = relationship(
        "TrainingRunSample",
        back_populates="training_run",
        cascade="all, delete-orphan"
    )
    generated_samples: Mapped[List["GeneratedSample"]] = relationship(
        "GeneratedSample",
        back_populates="training_run",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index('idx_training_runs_preset', 'preset_id'),
        Index('idx_training_runs_status', 'status'),
        Index('idx_training_runs_started', 'started_at'),
    )

    def get_config_snapshot(self) -> Dict[str, Any]:
        """Get the config snapshot as a dictionary."""
        return json.loads(self.config_snapshot_json)

    def set_config_snapshot(self, config: Dict[str, Any]) -> None:
        """Set the config snapshot from a dictionary."""
        self.config_snapshot_json = json.dumps(config)

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics as a dictionary."""
        if not self.metrics_json:
            return None
        return json.loads(self.metrics_json)

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set metrics from a dictionary."""
        self.metrics_json = json.dumps(metrics)

    def update_progress(
        self,
        epoch: int,
        step: int,
        loss: Optional[float] = None,
        smooth_loss: Optional[float] = None
    ) -> None:
        """Update training progress."""
        self.current_epoch = epoch
        self.current_step = step
        if loss is not None:
            self.final_loss = loss
        if smooth_loss is not None:
            self.final_smooth_loss = smooth_loss

    def start(self) -> None:
        """Mark training as started."""
        self.status = 'training'
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark training as completed."""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.total_duration_seconds = int(delta.total_seconds())

    def fail(self, error_message: str, traceback: Optional[str] = None) -> None:
        """Mark training as failed."""
        self.status = 'error'
        self.error_message = error_message
        self.error_traceback = traceback
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.total_duration_seconds = int(delta.total_seconds())

    def cancel(self) -> None:
        """Mark training as cancelled."""
        self.status = 'cancelled'
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.total_duration_seconds = int(delta.total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        """Convert training run to dictionary for API responses."""
        return {
            'id': self.id,
            'preset_id': self.preset_id,
            'preset_name': self.preset_name,
            'run_name': self.run_name,
            'workspace_dir': self.workspace_dir,
            'output_model_destination': self.output_model_destination,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'metrics': self.get_metrics(),
            'final_loss': self.final_loss,
            'final_smooth_loss': self.final_smooth_loss,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_duration_seconds': self.total_duration_seconds,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class TrainingRunConcept(Base):
    """Junction table linking training runs to concepts."""

    __tablename__ = 'training_run_concepts'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('training_runs.id', ondelete='CASCADE'),
        nullable=False
    )
    concept_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('concepts.id', ondelete='SET NULL'),
        nullable=True
    )

    # Snapshot of concept config at run time
    concept_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Order in the training
    position: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    training_run: Mapped["TrainingRun"] = relationship(
        "TrainingRun",
        back_populates="concepts"
    )

    __table_args__ = (
        Index('idx_run_concepts_run', 'training_run_id'),
        Index('idx_run_concepts_concept', 'concept_id'),
    )

    def get_concept_snapshot(self) -> Dict[str, Any]:
        """Get the concept snapshot as a dictionary."""
        return json.loads(self.concept_snapshot_json)

    def set_concept_snapshot(self, config: Dict[str, Any]) -> None:
        """Set the concept snapshot from a dictionary."""
        self.concept_snapshot_json = json.dumps(config)


class TrainingRunSample(Base):
    """Junction table linking training runs to samples."""

    __tablename__ = 'training_run_samples'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('training_runs.id', ondelete='CASCADE'),
        nullable=False
    )
    sample_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey('samples.id', ondelete='SET NULL'),
        nullable=True
    )

    # Snapshot of sample config at run time
    sample_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Order in the sampling
    position: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    training_run: Mapped["TrainingRun"] = relationship(
        "TrainingRun",
        back_populates="samples"
    )

    __table_args__ = (
        Index('idx_run_samples_run', 'training_run_id'),
        Index('idx_run_samples_sample', 'sample_id'),
    )

    def get_sample_snapshot(self) -> Dict[str, Any]:
        """Get the sample snapshot as a dictionary."""
        return json.loads(self.sample_snapshot_json)

    def set_sample_snapshot(self, config: Dict[str, Any]) -> None:
        """Set the sample snapshot from a dictionary."""
        self.sample_snapshot_json = json.dumps(config)

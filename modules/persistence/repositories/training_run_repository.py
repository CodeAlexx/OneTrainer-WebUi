"""Training run repository for session history management."""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, desc, and_

from ..models.training_run import TrainingRun, TrainingRunConcept, TrainingRunSample
from ..models.generated_sample import GeneratedSample
from .base import BaseRepository


class TrainingRunRepository(BaseRepository[TrainingRun]):
    """Repository for TrainingRun entities."""

    model_class = TrainingRun
    entity_type = 'training_run'

    def get_by_status(self, status: str) -> List[TrainingRun]:
        """Get all training runs with a specific status."""
        query = select(TrainingRun).where(TrainingRun.status == status)
        query = query.order_by(desc(TrainingRun.created_at))
        return list(self.session.execute(query).scalars().all())

    def get_active(self) -> List[TrainingRun]:
        """Get all active (non-terminal) training runs."""
        active_statuses = ['pending', 'starting', 'training', 'stopping']
        query = select(TrainingRun).where(TrainingRun.status.in_(active_statuses))
        query = query.order_by(desc(TrainingRun.created_at))
        return list(self.session.execute(query).scalars().all())

    def get_completed(self, limit: Optional[int] = None) -> List[TrainingRun]:
        """Get completed training runs."""
        query = select(TrainingRun).where(TrainingRun.status == 'completed')
        query = query.order_by(desc(TrainingRun.completed_at))
        if limit:
            query = query.limit(limit)
        return list(self.session.execute(query).scalars().all())

    def get_by_preset(self, preset_id: int) -> List[TrainingRun]:
        """Get all training runs for a specific preset."""
        query = select(TrainingRun).where(TrainingRun.preset_id == preset_id)
        query = query.order_by(desc(TrainingRun.created_at))
        return list(self.session.execute(query).scalars().all())

    def get_recent(self, limit: int = 10) -> List[TrainingRun]:
        """Get most recent training runs."""
        query = select(TrainingRun).order_by(desc(TrainingRun.created_at)).limit(limit)
        return list(self.session.execute(query).scalars().all())

    def create_run(
        self,
        config_snapshot: Dict[str, Any],
        preset_id: Optional[int] = None,
        preset_name: Optional[str] = None,
        run_name: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        output_model_destination: Optional[str] = None,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
        created_by: Optional[str] = None
    ) -> TrainingRun:
        """Create a new training run."""
        run = TrainingRun(
            preset_id=preset_id,
            preset_name=preset_name,
            run_name=run_name,
            workspace_dir=workspace_dir,
            output_model_destination=output_model_destination,
            config_snapshot_json=json.dumps(config_snapshot),
            status='pending',
            total_epochs=total_epochs,
            total_steps=total_steps
        )
        return self.create(run, created_by=created_by)

    def add_concept(
        self,
        run_id: int,
        concept_snapshot: Dict[str, Any],
        concept_id: Optional[int] = None,
        position: int = 0
    ) -> TrainingRunConcept:
        """Add a concept to a training run."""
        run_concept = TrainingRunConcept(
            training_run_id=run_id,
            concept_id=concept_id,
            concept_snapshot_json=json.dumps(concept_snapshot),
            position=position
        )
        self.session.add(run_concept)
        self.session.flush()
        return run_concept

    def add_sample(
        self,
        run_id: int,
        sample_snapshot: Dict[str, Any],
        sample_id: Optional[int] = None,
        position: int = 0
    ) -> TrainingRunSample:
        """Add a sample to a training run."""
        run_sample = TrainingRunSample(
            training_run_id=run_id,
            sample_id=sample_id,
            sample_snapshot_json=json.dumps(sample_snapshot),
            position=position
        )
        self.session.add(run_sample)
        self.session.flush()
        return run_sample

    def add_generated_sample(
        self,
        run_id: int,
        epoch: int,
        global_step: int,
        file_path: str,
        file_name: str,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs
    ) -> GeneratedSample:
        """Add a generated sample to a training run."""
        generated = GeneratedSample.from_file(
            training_run_id=run_id,
            epoch=epoch,
            global_step=global_step,
            file_path=file_path,
            file_name=file_name,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            **kwargs
        )
        self.session.add(generated)
        self.session.flush()
        return generated

    def start_run(self, run_id: int) -> TrainingRun:
        """Mark a training run as started."""
        run = self.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")
        run.start()
        self.session.flush()
        return run

    def update_progress(
        self,
        run_id: int,
        epoch: int,
        step: int,
        loss: Optional[float] = None,
        smooth_loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> TrainingRun:
        """Update training progress."""
        run = self.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")

        run.update_progress(epoch, step, loss, smooth_loss)
        if metrics:
            run.set_metrics(metrics)
        self.session.flush()
        return run

    def complete_run(self, run_id: int) -> TrainingRun:
        """Mark a training run as completed."""
        run = self.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")
        run.complete()
        self.session.flush()
        return run

    def fail_run(
        self,
        run_id: int,
        error_message: str,
        traceback: Optional[str] = None
    ) -> TrainingRun:
        """Mark a training run as failed."""
        run = self.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")
        run.fail(error_message, traceback)
        self.session.flush()
        return run

    def cancel_run(self, run_id: int) -> TrainingRun:
        """Mark a training run as cancelled."""
        run = self.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")
        run.cancel()
        self.session.flush()
        return run

    def get_generated_samples(
        self,
        run_id: int,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> List[GeneratedSample]:
        """Get generated samples for a training run."""
        query = select(GeneratedSample).where(
            GeneratedSample.training_run_id == run_id
        )
        if epoch is not None:
            query = query.where(GeneratedSample.epoch == epoch)
        if step is not None:
            query = query.where(GeneratedSample.global_step == step)
        query = query.order_by(GeneratedSample.epoch, GeneratedSample.global_step)
        return list(self.session.execute(query).scalars().all())

    def get_run_concepts(self, run_id: int) -> List[TrainingRunConcept]:
        """Get all concepts for a training run."""
        query = (
            select(TrainingRunConcept)
            .where(TrainingRunConcept.training_run_id == run_id)
            .order_by(TrainingRunConcept.position)
        )
        return list(self.session.execute(query).scalars().all())

    def get_run_samples(self, run_id: int) -> List[TrainingRunSample]:
        """Get all samples for a training run."""
        query = (
            select(TrainingRunSample)
            .where(TrainingRunSample.training_run_id == run_id)
            .order_by(TrainingRunSample.position)
        )
        return list(self.session.execute(query).scalars().all())

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall training statistics."""
        from sqlalchemy import func

        # Count by status
        status_counts = {}
        for status in ['pending', 'starting', 'training', 'stopping', 'completed', 'error', 'cancelled']:
            query = select(func.count(TrainingRun.id)).where(TrainingRun.status == status)
            status_counts[status] = self.session.execute(query).scalar() or 0

        # Total runs
        total_query = select(func.count(TrainingRun.id))
        total = self.session.execute(total_query).scalar() or 0

        # Average duration for completed runs
        avg_duration_query = select(func.avg(TrainingRun.total_duration_seconds)).where(
            TrainingRun.status == 'completed'
        )
        avg_duration = self.session.execute(avg_duration_query).scalar()

        return {
            'total_runs': total,
            'status_counts': status_counts,
            'average_duration_seconds': avg_duration,
        }

    def _entity_to_dict(self, entity: TrainingRun) -> Dict[str, Any]:
        """Convert training run to dictionary for versioning."""
        return entity.to_dict()

    def _restore_entity_from_dict(self, entity: TrainingRun, data: Dict[str, Any]) -> None:
        """Restore training run from dictionary data."""
        entity.preset_id = data.get('preset_id')
        entity.preset_name = data.get('preset_name')
        entity.run_name = data.get('run_name')
        entity.workspace_dir = data.get('workspace_dir')
        entity.output_model_destination = data.get('output_model_destination')
        entity.status = data.get('status', 'pending')
        entity.current_epoch = data.get('current_epoch', 0)
        entity.total_epochs = data.get('total_epochs')
        entity.current_step = data.get('current_step', 0)
        entity.total_steps = data.get('total_steps')
        entity.final_loss = data.get('final_loss')
        entity.final_smooth_loss = data.get('final_smooth_loss')
        entity.error_message = data.get('error_message')

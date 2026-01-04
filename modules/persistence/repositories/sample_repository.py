"""Sample repository for inference configuration management."""

import json
from typing import Optional, List, Dict, Any

from sqlalchemy import select

from ..models.sample import Sample
from .base import BaseRepository


class SampleRepository(BaseRepository[Sample]):
    """Repository for Sample entities."""

    model_class = Sample
    entity_type = 'sample'

    def get_by_name(self, name: str, include_deleted: bool = False) -> Optional[Sample]:
        """Get sample by name."""
        query = select(Sample).where(Sample.name == name)
        if not include_deleted:
            query = query.where(Sample.deleted_at.is_(None))
        return self.session.execute(query).scalar_one_or_none()

    def get_enabled(self, include_deleted: bool = False) -> List[Sample]:
        """Get all enabled samples."""
        query = select(Sample).where(Sample.enabled == True)
        if not include_deleted:
            query = query.where(Sample.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def search_by_prompt(
        self,
        query_str: str,
        include_deleted: bool = False
    ) -> List[Sample]:
        """Search samples by prompt text."""
        search_pattern = f"%{query_str}%"
        query = select(Sample).where(Sample.prompt.ilike(search_pattern))
        if not include_deleted:
            query = query.where(Sample.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def create_from_dict(
        self,
        config_dict: Dict[str, Any],
        name: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Sample:
        """Create a sample from a config dictionary."""
        sample = Sample.from_dict(config_dict, name=name)
        return self.create(sample, created_by=created_by)

    def create_from_sample_config(
        self,
        config: Any,  # SampleConfig
        name: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Sample:
        """Create a sample from a SampleConfig object."""
        sample = Sample.from_sample_config(config, name=name)
        return self.create(sample, created_by=created_by)

    def update_config(
        self,
        sample_id: int,
        config_dict: Dict[str, Any],
        change_description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Sample:
        """Update sample configuration."""
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        # Track changed fields
        old_config = sample.get_config_dict()
        changed_fields = self._find_changed_fields(old_config, config_dict)

        # Update sample
        sample.set_config_dict(config_dict)
        sample.prompt = config_dict.get('prompt', sample.prompt)
        sample.negative_prompt = config_dict.get('negative_prompt')
        sample.width = config_dict.get('width', sample.width)
        sample.height = config_dict.get('height', sample.height)
        sample.seed = config_dict.get('seed', sample.seed)
        sample.random_seed = config_dict.get('random_seed', sample.random_seed)
        sample.diffusion_steps = config_dict.get('diffusion_steps', sample.diffusion_steps)
        sample.cfg_scale = config_dict.get('cfg_scale', sample.cfg_scale)
        sample.noise_scheduler = config_dict.get('noise_scheduler')
        sample.frames = config_dict.get('frames', sample.frames)
        sample.length = config_dict.get('length', sample.length)
        sample.enabled = config_dict.get('enabled', sample.enabled)

        return self.update(
            sample,
            changed_fields=changed_fields,
            change_description=change_description,
            created_by=created_by
        )

    def toggle_enabled(
        self,
        sample_id: int,
        created_by: Optional[str] = None
    ) -> Sample:
        """Toggle sample enabled status."""
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        sample.enabled = not sample.enabled
        return self.update(
            sample,
            changed_fields=['enabled'],
            change_description=f"{'Enabled' if sample.enabled else 'Disabled'} sample",
            created_by=created_by
        )

    def bulk_create(
        self,
        configs: List[Dict[str, Any]],
        name_prefix: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> List[Sample]:
        """Create multiple samples from a list of config dictionaries."""
        samples = []
        for i, config_dict in enumerate(configs):
            name = f"{name_prefix}_{i}" if name_prefix else None
            sample = self.create_from_dict(config_dict, name=name, created_by=created_by)
            samples.append(sample)
        return samples

    def _find_changed_fields(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> List[str]:
        """Find which fields changed between two configs."""
        changed = []
        all_keys = set(old_config.keys()) | set(new_config.keys())
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                changed.append(key)
        return changed

    def _restore_entity_from_dict(self, entity: Sample, data: Dict[str, Any]) -> None:
        """Restore sample from dictionary data."""
        entity.name = data.get('name')
        entity.prompt = data.get('prompt', entity.prompt)
        entity.negative_prompt = data.get('negative_prompt')
        entity.width = data.get('width', entity.width)
        entity.height = data.get('height', entity.height)
        entity.seed = data.get('seed', entity.seed)
        entity.random_seed = data.get('random_seed', False)
        entity.diffusion_steps = data.get('diffusion_steps', entity.diffusion_steps)
        entity.cfg_scale = data.get('cfg_scale', entity.cfg_scale)
        entity.noise_scheduler = data.get('noise_scheduler')
        entity.frames = data.get('frames', entity.frames)
        entity.length = data.get('length', entity.length)
        entity.enabled = data.get('enabled', True)
        if 'config' in data:
            entity.set_config_dict(data['config'])

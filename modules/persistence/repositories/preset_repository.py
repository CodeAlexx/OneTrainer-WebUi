"""Preset repository for training configuration management."""

import json
from typing import Optional, List, Dict, Any

from sqlalchemy import select, or_

from ..models.preset import Preset
from .base import BaseRepository


class PresetRepository(BaseRepository[Preset]):
    """Repository for Preset entities."""

    model_class = Preset
    entity_type = 'preset'

    def get_by_name(self, name: str, include_deleted: bool = False) -> Optional[Preset]:
        """Get preset by name."""
        query = select(Preset).where(Preset.name == name)
        if not include_deleted:
            query = query.where(Preset.deleted_at.is_(None))
        return self.session.execute(query).scalar_one_or_none()

    def get_by_model_type(
        self,
        model_type: str,
        include_deleted: bool = False
    ) -> List[Preset]:
        """Get all presets for a specific model type."""
        query = select(Preset).where(Preset.model_type == model_type)
        if not include_deleted:
            query = query.where(Preset.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def get_favorites(self, include_deleted: bool = False) -> List[Preset]:
        """Get all favorite presets."""
        query = select(Preset).where(Preset.is_favorite == True)
        if not include_deleted:
            query = query.where(Preset.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def get_builtin(self, include_deleted: bool = False) -> List[Preset]:
        """Get all builtin presets."""
        query = select(Preset).where(Preset.is_builtin == True)
        if not include_deleted:
            query = query.where(Preset.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def search(
        self,
        query_str: str,
        include_deleted: bool = False
    ) -> List[Preset]:
        """Search presets by name or description."""
        search_pattern = f"%{query_str}%"
        query = select(Preset).where(
            or_(
                Preset.name.ilike(search_pattern),
                Preset.description.ilike(search_pattern)
            )
        )
        if not include_deleted:
            query = query.where(Preset.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def create_from_dict(
        self,
        name: str,
        config_dict: Dict[str, Any],
        description: Optional[str] = None,
        is_builtin: bool = False,
        created_by: Optional[str] = None
    ) -> Preset:
        """Create a preset from a config dictionary."""
        preset = Preset.from_dict(
            config_dict=config_dict,
            name=name,
            description=description,
            is_builtin=is_builtin,
            created_by=created_by
        )
        return self.create(preset, created_by=created_by)

    def create_from_train_config(
        self,
        name: str,
        config: Any,  # TrainConfig
        description: Optional[str] = None,
        is_builtin: bool = False,
        created_by: Optional[str] = None
    ) -> Preset:
        """Create a preset from a TrainConfig object."""
        preset = Preset.from_train_config(
            config=config,
            name=name,
            description=description,
            is_builtin=is_builtin,
            created_by=created_by
        )
        return self.create(preset, created_by=created_by)

    def update_config(
        self,
        preset_id: int,
        config_dict: Dict[str, Any],
        change_description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Preset:
        """Update preset configuration."""
        preset = self.get_by_id(preset_id)
        if not preset:
            raise ValueError(f"Preset {preset_id} not found")

        # Track changed fields
        old_config = preset.get_config_dict()
        changed_fields = self._find_changed_fields(old_config, config_dict)

        # Update preset
        preset.set_config_dict(config_dict)
        preset.model_type = config_dict.get('model_type', preset.model_type)
        preset.training_method = config_dict.get('training_method', preset.training_method)
        preset.base_model_name = config_dict.get('base_model_name', preset.base_model_name)
        preset.peft_type = config_dict.get('peft_type', preset.peft_type)

        return self.update(
            preset,
            changed_fields=changed_fields,
            change_description=change_description,
            created_by=created_by
        )

    def toggle_favorite(
        self,
        preset_id: int,
        created_by: Optional[str] = None
    ) -> Preset:
        """Toggle preset favorite status."""
        preset = self.get_by_id(preset_id)
        if not preset:
            raise ValueError(f"Preset {preset_id} not found")

        preset.is_favorite = not preset.is_favorite
        return self.update(
            preset,
            changed_fields=['is_favorite'],
            change_description=f"{'Added to' if preset.is_favorite else 'Removed from'} favorites",
            created_by=created_by
        )

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

    def _restore_entity_from_dict(self, entity: Preset, data: Dict[str, Any]) -> None:
        """Restore preset from dictionary data."""
        entity.name = data.get('name', entity.name)
        entity.description = data.get('description')
        entity.model_type = data.get('model_type', entity.model_type)
        entity.training_method = data.get('training_method', entity.training_method)
        entity.base_model_name = data.get('base_model_name')
        entity.peft_type = data.get('peft_type')
        entity.is_builtin = data.get('is_builtin', False)
        entity.is_favorite = data.get('is_favorite', False)
        if 'tags' in data:
            entity.set_tags_list(data['tags'])
        if 'config' in data:
            entity.set_config_dict(data['config'])

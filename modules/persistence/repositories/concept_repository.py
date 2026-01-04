"""Concept repository for dataset definition management."""

import json
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_

from ..models.concept import Concept
from .base import BaseRepository


class ConceptRepository(BaseRepository[Concept]):
    """Repository for Concept entities."""

    model_class = Concept
    entity_type = 'concept'

    def get_by_name(self, name: str, include_deleted: bool = False) -> List[Concept]:
        """Get concepts by name (may have multiple with same name)."""
        query = select(Concept).where(Concept.name == name)
        if not include_deleted:
            query = query.where(Concept.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def get_by_name_and_path(
        self,
        name: str,
        path: str,
        include_deleted: bool = False
    ) -> Optional[Concept]:
        """Get a specific concept by name and path."""
        query = select(Concept).where(
            and_(Concept.name == name, Concept.path == path)
        )
        if not include_deleted:
            query = query.where(Concept.deleted_at.is_(None))
        return self.session.execute(query).scalar_one_or_none()

    def get_by_path(self, path: str, include_deleted: bool = False) -> List[Concept]:
        """Get all concepts for a specific path."""
        query = select(Concept).where(Concept.path == path)
        if not include_deleted:
            query = query.where(Concept.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def get_enabled(self, include_deleted: bool = False) -> List[Concept]:
        """Get all enabled concepts."""
        query = select(Concept).where(Concept.enabled == True)
        if not include_deleted:
            query = query.where(Concept.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def get_by_type(
        self,
        concept_type: str,
        include_deleted: bool = False
    ) -> List[Concept]:
        """Get all concepts of a specific type."""
        query = select(Concept).where(Concept.concept_type == concept_type)
        if not include_deleted:
            query = query.where(Concept.deleted_at.is_(None))
        return list(self.session.execute(query).scalars().all())

    def create_from_dict(
        self,
        config_dict: Dict[str, Any],
        created_by: Optional[str] = None
    ) -> Concept:
        """Create a concept from a config dictionary."""
        concept = Concept.from_dict(config_dict)
        return self.create(concept, created_by=created_by)

    def create_from_concept_config(
        self,
        config: Any,  # ConceptConfig
        created_by: Optional[str] = None
    ) -> Concept:
        """Create a concept from a ConceptConfig object."""
        concept = Concept.from_concept_config(config)
        return self.create(concept, created_by=created_by)

    def update_config(
        self,
        concept_id: int,
        config_dict: Dict[str, Any],
        change_description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Concept:
        """Update concept configuration."""
        concept = self.get_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept {concept_id} not found")

        # Track changed fields
        old_config = concept.get_config_dict()
        changed_fields = self._find_changed_fields(old_config, config_dict)

        # Update concept
        concept.set_config_dict(config_dict)
        concept.name = config_dict.get('name', concept.name)
        concept.path = config_dict.get('path', concept.path)
        concept.concept_type = config_dict.get('type', concept.concept_type)
        concept.enabled = config_dict.get('enabled', concept.enabled)
        concept.seed = config_dict.get('seed')
        concept.include_subdirectories = config_dict.get('include_subdirectories', False)
        concept.image_variations = config_dict.get('image_variations', 1)
        concept.text_variations = config_dict.get('text_variations', 1)
        concept.balancing = config_dict.get('balancing', 1.0)
        concept.balancing_strategy = config_dict.get('balancing_strategy', 'REPEATS')
        concept.loss_weight = config_dict.get('loss_weight', 1.0)

        if 'image' in config_dict:
            concept.set_image_config(config_dict['image'])
        if 'text' in config_dict:
            concept.set_text_config(config_dict['text'])
        if 'concept_stats' in config_dict:
            concept.set_stats(config_dict['concept_stats'])

        return self.update(
            concept,
            changed_fields=changed_fields,
            change_description=change_description,
            created_by=created_by
        )

    def update_stats(
        self,
        concept_id: int,
        stats: Dict[str, Any],
        created_by: Optional[str] = None
    ) -> Concept:
        """Update concept statistics."""
        concept = self.get_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept {concept_id} not found")

        concept.set_stats(stats)
        return self.update(
            concept,
            changed_fields=['concept_stats'],
            change_description='Updated concept statistics',
            created_by=created_by
        )

    def toggle_enabled(
        self,
        concept_id: int,
        created_by: Optional[str] = None
    ) -> Concept:
        """Toggle concept enabled status."""
        concept = self.get_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept {concept_id} not found")

        concept.enabled = not concept.enabled
        return self.update(
            concept,
            changed_fields=['enabled'],
            change_description=f"{'Enabled' if concept.enabled else 'Disabled'} concept",
            created_by=created_by
        )

    def bulk_create(
        self,
        configs: List[Dict[str, Any]],
        created_by: Optional[str] = None
    ) -> List[Concept]:
        """Create multiple concepts from a list of config dictionaries."""
        concepts = []
        for config_dict in configs:
            concept = self.create_from_dict(config_dict, created_by=created_by)
            concepts.append(concept)
        return concepts

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

    def _restore_entity_from_dict(self, entity: Concept, data: Dict[str, Any]) -> None:
        """Restore concept from dictionary data."""
        entity.name = data.get('name', entity.name)
        entity.path = data.get('path', entity.path)
        entity.concept_type = data.get('concept_type', entity.concept_type)
        entity.enabled = data.get('enabled', True)
        entity.seed = data.get('seed')
        entity.include_subdirectories = data.get('include_subdirectories', False)
        entity.image_variations = data.get('image_variations', 1)
        entity.text_variations = data.get('text_variations', 1)
        entity.balancing = data.get('balancing', 1.0)
        entity.balancing_strategy = data.get('balancing_strategy', 'REPEATS')
        entity.loss_weight = data.get('loss_weight', 1.0)
        if 'image_config' in data:
            entity.set_image_config(data['image_config'])
        if 'text_config' in data:
            entity.set_text_config(data['text_config'])
        if 'stats' in data and data['stats']:
            entity.set_stats(data['stats'])
        if 'config' in data:
            entity.set_config_dict(data['config'])

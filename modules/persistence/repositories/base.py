"""Base repository with common CRUD operations."""

from datetime import datetime
from typing import TypeVar, Generic, List, Optional, Type, Dict, Any

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from ..models.base import Base, SoftDeleteMixin
from ..models.entity_version import EntityVersion

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations and audit trail."""

    model_class: Type[T]
    entity_type: str  # For versioning: 'preset', 'concept', 'sample', etc.

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, id: int, include_deleted: bool = False) -> Optional[T]:
        """Get entity by ID."""
        query = select(self.model_class).where(self.model_class.id == id)
        if not include_deleted and issubclass(self.model_class, SoftDeleteMixin):
            query = query.where(self.model_class.deleted_at.is_(None))
        return self.session.execute(query).scalar_one_or_none()

    def get_all(self, include_deleted: bool = False) -> List[T]:
        """Get all entities."""
        query = select(self.model_class)
        if not include_deleted and issubclass(self.model_class, SoftDeleteMixin):
            query = query.where(self.model_class.deleted_at.is_(None))
        query = query.order_by(desc(self.model_class.id))
        return list(self.session.execute(query).scalars().all())

    def create(self, entity: T, created_by: Optional[str] = None) -> T:
        """Create a new entity with version tracking."""
        self.session.add(entity)
        self.session.flush()  # Get the ID

        # Create version entry
        self._create_version(
            entity_id=entity.id,
            data=self._entity_to_dict(entity),
            change_type='create',
            created_by=created_by
        )

        return entity

    def update(
        self,
        entity: T,
        changed_fields: Optional[List[str]] = None,
        change_description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> T:
        """Update an existing entity with version tracking."""
        self.session.merge(entity)
        self.session.flush()

        # Create version entry
        self._create_version(
            entity_id=entity.id,
            data=self._entity_to_dict(entity),
            change_type='update',
            change_description=change_description,
            changed_fields=changed_fields,
            created_by=created_by
        )

        return entity

    def delete(
        self,
        entity: T,
        soft: bool = True,
        created_by: Optional[str] = None
    ) -> None:
        """Delete an entity (soft or hard) with version tracking."""
        if soft and isinstance(entity, SoftDeleteMixin):
            entity.soft_delete()
            self.session.merge(entity)
            change_type = 'delete'
        else:
            self.session.delete(entity)
            change_type = 'delete'

        self.session.flush()

        # Create version entry (for soft delete, record the state)
        if soft and isinstance(entity, SoftDeleteMixin):
            self._create_version(
                entity_id=entity.id,
                data=self._entity_to_dict(entity),
                change_type=change_type,
                created_by=created_by
            )

    def restore(self, entity: T, created_by: Optional[str] = None) -> T:
        """Restore a soft-deleted entity with version tracking."""
        if isinstance(entity, SoftDeleteMixin):
            entity.restore()
            self.session.merge(entity)
            self.session.flush()

            # Create version entry
            self._create_version(
                entity_id=entity.id,
                data=self._entity_to_dict(entity),
                change_type='restore',
                created_by=created_by
            )

        return entity

    def get_version_history(
        self,
        entity_id: int,
        limit: Optional[int] = None
    ) -> List[EntityVersion]:
        """Get version history for an entity."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == self.entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .order_by(desc(EntityVersion.version))
        )
        if limit:
            query = query.limit(limit)
        return list(self.session.execute(query).scalars().all())

    def get_version(self, entity_id: int, version: int) -> Optional[EntityVersion]:
        """Get a specific version of an entity."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == self.entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .where(EntityVersion.version == version)
        )
        return self.session.execute(query).scalar_one_or_none()

    def rollback_to_version(
        self,
        entity_id: int,
        target_version: int,
        created_by: Optional[str] = None
    ) -> T:
        """Rollback an entity to a specific version."""
        version = self.get_version(entity_id, target_version)
        if not version:
            raise ValueError(f"Version {target_version} not found for {self.entity_type} {entity_id}")

        # Get current entity
        entity = self.get_by_id(entity_id, include_deleted=True)
        if not entity:
            raise ValueError(f"{self.entity_type} {entity_id} not found")

        # Restore from version data
        data = version.get_data()
        self._restore_entity_from_dict(entity, data)

        # If entity was soft-deleted, restore it
        if isinstance(entity, SoftDeleteMixin) and entity.is_deleted:
            entity.restore()

        self.session.merge(entity)
        self.session.flush()

        # Create version entry for the rollback
        self._create_version(
            entity_id=entity_id,
            data=self._entity_to_dict(entity),
            change_type='restore',
            change_description=f'Rolled back to version {target_version}',
            created_by=created_by
        )

        return entity

    def _create_version(
        self,
        entity_id: int,
        data: Dict[str, Any],
        change_type: str,
        change_description: Optional[str] = None,
        changed_fields: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> EntityVersion:
        """Create a new version entry."""
        # Get next version number
        current_version = self._get_latest_version_number(entity_id)
        next_version = (current_version or 0) + 1

        version = EntityVersion.create_version(
            entity_type=self.entity_type,
            entity_id=entity_id,
            version=next_version,
            data=data,
            change_type=change_type,
            change_description=change_description,
            changed_fields=changed_fields,
            created_by=created_by
        )
        self.session.add(version)
        self.session.flush()
        return version

    def _get_latest_version_number(self, entity_id: int) -> Optional[int]:
        """Get the latest version number for an entity."""
        query = (
            select(EntityVersion.version)
            .where(EntityVersion.entity_type == self.entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .order_by(desc(EntityVersion.version))
            .limit(1)
        )
        return self.session.execute(query).scalar_one_or_none()

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for versioning. Override in subclasses."""
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        raise NotImplementedError("Entity must have to_dict method or override _entity_to_dict")

    def _restore_entity_from_dict(self, entity: T, data: Dict[str, Any]) -> None:
        """Restore entity from dictionary. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _restore_entity_from_dict")

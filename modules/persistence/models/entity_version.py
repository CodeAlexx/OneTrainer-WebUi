"""Entity version model for audit trail."""

import json
from datetime import datetime
from typing import Optional, List, Any, Dict

from sqlalchemy import String, Text, Integer, Index, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class EntityVersion(Base):
    """Polymorphic audit table for all entities."""

    __tablename__ = 'entity_versions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Polymorphic reference
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Version info
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # Snapshot of entity at this version
    data_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Change tracking
    change_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'create', 'update', 'delete', 'restore'
    change_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    changed_fields: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        UniqueConstraint('entity_type', 'entity_id', 'version', name='uq_entity_version'),
        Index('idx_versions_entity', 'entity_type', 'entity_id'),
        Index('idx_versions_created', 'created_at'),
    )

    def get_data(self) -> Dict[str, Any]:
        """Get the snapshot data as a dictionary."""
        return json.loads(self.data_json)

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the snapshot data from a dictionary."""
        self.data_json = json.dumps(data)

    def get_changed_fields_list(self) -> List[str]:
        """Get changed fields as a list."""
        if not self.changed_fields:
            return []
        return json.loads(self.changed_fields)

    def set_changed_fields_list(self, fields: List[str]) -> None:
        """Set changed fields from a list."""
        self.changed_fields = json.dumps(fields)

    @classmethod
    def create_version(
        cls,
        entity_type: str,
        entity_id: int,
        version: int,
        data: Dict[str, Any],
        change_type: str,
        change_description: Optional[str] = None,
        changed_fields: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> "EntityVersion":
        """Create a new entity version."""
        return cls(
            entity_type=entity_type,
            entity_id=entity_id,
            version=version,
            data_json=json.dumps(data),
            change_type=change_type,
            change_description=change_description,
            changed_fields=json.dumps(changed_fields) if changed_fields else None,
            created_by=created_by
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity version to dictionary for API responses."""
        return {
            'id': self.id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'version': self.version,
            'data': self.get_data(),
            'change_type': self.change_type,
            'change_description': self.change_description,
            'changed_fields': self.get_changed_fields_list(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
        }

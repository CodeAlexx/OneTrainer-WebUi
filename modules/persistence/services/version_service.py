"""Version service for audit trail management."""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from ..models.entity_version import EntityVersion


class VersionService:
    """Service for managing entity version history and audit trail."""

    def __init__(self, session: Session):
        self.session = session

    def get_version_history(
        self,
        entity_type: str,
        entity_id: int,
        limit: Optional[int] = None
    ) -> List[EntityVersion]:
        """Get version history for an entity."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .order_by(desc(EntityVersion.version))
        )
        if limit:
            query = query.limit(limit)
        return list(self.session.execute(query).scalars().all())

    def get_version(
        self,
        entity_type: str,
        entity_id: int,
        version: int
    ) -> Optional[EntityVersion]:
        """Get a specific version of an entity."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .where(EntityVersion.version == version)
        )
        return self.session.execute(query).scalar_one_or_none()

    def get_latest_version(
        self,
        entity_type: str,
        entity_id: int
    ) -> Optional[EntityVersion]:
        """Get the latest version of an entity."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .order_by(desc(EntityVersion.version))
            .limit(1)
        )
        return self.session.execute(query).scalar_one_or_none()

    def get_versions_by_date_range(
        self,
        entity_type: str,
        entity_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[EntityVersion]:
        """Get versions within a date range."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .where(EntityVersion.created_at >= start_date)
            .where(EntityVersion.created_at <= end_date)
            .order_by(desc(EntityVersion.version))
        )
        return list(self.session.execute(query).scalars().all())

    def get_versions_by_change_type(
        self,
        entity_type: str,
        entity_id: int,
        change_type: str
    ) -> List[EntityVersion]:
        """Get versions by change type (create, update, delete, restore)."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.entity_type == entity_type)
            .where(EntityVersion.entity_id == entity_id)
            .where(EntityVersion.change_type == change_type)
            .order_by(desc(EntityVersion.version))
        )
        return list(self.session.execute(query).scalars().all())

    def get_recent_changes(
        self,
        limit: int = 50,
        entity_types: Optional[List[str]] = None
    ) -> List[EntityVersion]:
        """Get recent changes across all entities."""
        query = select(EntityVersion).order_by(desc(EntityVersion.created_at))
        if entity_types:
            query = query.where(EntityVersion.entity_type.in_(entity_types))
        query = query.limit(limit)
        return list(self.session.execute(query).scalars().all())

    def get_changes_by_user(
        self,
        created_by: str,
        limit: Optional[int] = None
    ) -> List[EntityVersion]:
        """Get all changes made by a specific user."""
        query = (
            select(EntityVersion)
            .where(EntityVersion.created_by == created_by)
            .order_by(desc(EntityVersion.created_at))
        )
        if limit:
            query = query.limit(limit)
        return list(self.session.execute(query).scalars().all())

    def compare_versions(
        self,
        entity_type: str,
        entity_id: int,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Compare two versions and return the differences."""
        v1 = self.get_version(entity_type, entity_id, version1)
        v2 = self.get_version(entity_type, entity_id, version2)

        if not v1 or not v2:
            raise ValueError(f"One or both versions not found")

        data1 = v1.get_data()
        data2 = v2.get_data()

        # Find differences
        added = {}
        removed = {}
        changed = {}

        all_keys = set(data1.keys()) | set(data2.keys())
        for key in all_keys:
            if key not in data1:
                added[key] = data2[key]
            elif key not in data2:
                removed[key] = data1[key]
            elif data1[key] != data2[key]:
                changed[key] = {
                    'from': data1[key],
                    'to': data2[key]
                }

        return {
            'version1': version1,
            'version2': version2,
            'added': added,
            'removed': removed,
            'changed': changed,
        }

    def get_entity_audit_summary(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """Get an audit summary for an entity."""
        versions = self.get_version_history(entity_type, entity_id)

        if not versions:
            return {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'total_versions': 0,
                'first_created': None,
                'last_modified': None,
                'change_counts': {},
            }

        change_counts = {}
        for v in versions:
            change_counts[v.change_type] = change_counts.get(v.change_type, 0) + 1

        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'total_versions': len(versions),
            'first_created': versions[-1].created_at.isoformat() if versions else None,
            'last_modified': versions[0].created_at.isoformat() if versions else None,
            'change_counts': change_counts,
            'contributors': list(set(v.created_by for v in versions if v.created_by)),
        }

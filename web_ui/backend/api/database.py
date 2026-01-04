"""
Database management REST API endpoints.

Provides endpoints for managing database persistence layer:
- Initialize database
- Migrate JSON files to database
- Export database to JSON
- Manage presets, concepts, samples via database
- Version history and rollback
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status, Query

from web_ui.backend.models import CommandResponse

# Lazy import to handle case where SQLAlchemy is not installed
def get_db_session():
    """Get database session with lazy import."""
    try:
        from modules.persistence.database import get_session
        return get_session()
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database module not available. Install SQLAlchemy: pip install SQLAlchemy alembic"
        )

def get_db_enabled():
    """Check if database is enabled."""
    try:
        from modules.persistence.config import DatabaseConfig
        return DatabaseConfig.is_db_enabled()
    except ImportError:
        return False


router = APIRouter()


# ============================================================================
# Database Management
# ============================================================================

@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Get database status",
)
async def get_db_status() -> Dict[str, Any]:
    """Get database status and statistics."""
    try:
        from modules.persistence.config import DatabaseConfig
        from modules.persistence.database import get_session
        from modules.persistence.repositories import (
            PresetRepository, ConceptRepository,
            SampleRepository, TrainingRunRepository
        )

        db_path = DatabaseConfig.get_db_path()
        db_exists = db_path.exists()

        if not db_exists:
            return {
                "enabled": DatabaseConfig.is_db_enabled(),
                "initialized": False,
                "path": str(db_path),
                "counts": None
            }

        with get_session() as session:
            preset_repo = PresetRepository(session)
            concept_repo = ConceptRepository(session)
            sample_repo = SampleRepository(session)
            run_repo = TrainingRunRepository(session)

            return {
                "enabled": DatabaseConfig.is_db_enabled(),
                "initialized": True,
                "path": str(db_path),
                "counts": {
                    "presets": len(preset_repo.get_all()),
                    "concepts": len(concept_repo.get_all()),
                    "samples": len(sample_repo.get_all()),
                    "training_runs": run_repo.get_statistics()["total_runs"]
                }
            }

    except ImportError:
        return {
            "enabled": False,
            "initialized": False,
            "path": None,
            "error": "Database module not installed"
        }
    except Exception as e:
        return {
            "enabled": False,
            "initialized": False,
            "path": None,
            "error": str(e)
        }


@router.post(
    "/init",
    response_model=CommandResponse,
    summary="Initialize database",
)
async def init_database() -> CommandResponse:
    """Initialize the database (create tables)."""
    try:
        from modules.persistence.database import init_database as do_init
        from modules.persistence.config import DatabaseConfig

        do_init()
        return CommandResponse(
            success=True,
            message=f"Database initialized at {DatabaseConfig.get_db_path()}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize database: {str(e)}"
        )


@router.post(
    "/migrate",
    response_model=Dict[str, Any],
    summary="Migrate JSON files to database",
)
async def migrate_json_to_db(
    include_presets: bool = Query(True, description="Migrate presets"),
    include_concepts: bool = Query(True, description="Migrate concepts"),
    include_samples: bool = Query(True, description="Migrate samples"),
) -> Dict[str, Any]:
    """Migrate JSON files to database."""
    try:
        from modules.persistence.database import get_session, init_database
        from modules.persistence.services.migration_service import MigrationService

        # Ensure database is initialized
        init_database()

        import os
        onetrainer_root = Path(os.environ.get(
            "ONETRAINER_ROOT",
            Path(__file__).parent.parent.parent.parent
        ))

        with get_session() as session:
            service = MigrationService(session)

            results = {}

            if include_presets:
                results['presets'] = service.migrate_presets(
                    onetrainer_root / 'training_presets',
                    created_by='web_ui'
                )

            if include_concepts:
                results['concepts'] = service.migrate_concepts(
                    onetrainer_root / 'training_concepts',
                    created_by='web_ui'
                )

            if include_samples:
                results['samples'] = service.migrate_samples(
                    onetrainer_root / 'training_samples',
                    created_by='web_ui'
                )

            session.commit()

        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )


# ============================================================================
# Presets from Database
# ============================================================================

@router.get(
    "/presets",
    response_model=List[Dict[str, Any]],
    summary="List presets from database",
)
async def list_db_presets(
    include_builtin: bool = Query(True, description="Include builtin presets"),
    favorites_only: bool = Query(False, description="Only return favorites"),
) -> List[Dict[str, Any]]:
    """List all presets from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)

            if favorites_only:
                presets = repo.get_favorites()
            else:
                presets = repo.get_all()

            if not include_builtin:
                presets = [p for p in presets if not p.is_builtin]

            return [p.to_dict() for p in presets]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list presets: {str(e)}"
        )


@router.get(
    "/presets/{preset_id}",
    response_model=Dict[str, Any],
    summary="Get preset by ID",
)
async def get_db_preset(preset_id: int) -> Dict[str, Any]:
    """Get a preset by ID."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)
            preset = repo.get_by_id(preset_id)

            if not preset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Preset {preset_id} not found"
                )

            return preset.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preset: {str(e)}"
        )


@router.post(
    "/presets",
    response_model=Dict[str, Any],
    summary="Create preset in database",
)
async def create_db_preset(
    name: str = Query(..., description="Preset name"),
    config: Dict[str, Any] = None,
    description: Optional[str] = Query(None, description="Preset description"),
) -> Dict[str, Any]:
    """Create a new preset in database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)

            # Check if name already exists
            existing = repo.get_by_name(name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Preset '{name}' already exists"
                )

            preset = repo.create_from_dict(
                name=name,
                config_dict=config or {},
                description=description,
                created_by='web_ui'
            )
            session.commit()

            return preset.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create preset: {str(e)}"
        )


@router.put(
    "/presets/{preset_id}",
    response_model=Dict[str, Any],
    summary="Update preset in database",
)
async def update_db_preset(
    preset_id: int,
    config: Dict[str, Any] = None,
    description: Optional[str] = Query(None, description="Update description"),
) -> Dict[str, Any]:
    """Update a preset in database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)

            preset = repo.get_by_id(preset_id)
            if not preset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Preset {preset_id} not found"
                )

            if config:
                preset = repo.update_config(
                    preset_id=preset_id,
                    config_dict=config,
                    change_description=description,
                    created_by='web_ui'
                )

            session.commit()
            return preset.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preset: {str(e)}"
        )


@router.delete(
    "/presets/{preset_id}",
    response_model=CommandResponse,
    summary="Delete preset from database",
)
async def delete_db_preset(
    preset_id: int,
    hard: bool = Query(False, description="Hard delete (permanent)"),
) -> CommandResponse:
    """Delete a preset from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)

            preset = repo.get_by_id(preset_id, include_deleted=True)
            if not preset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Preset {preset_id} not found"
                )

            name = preset.name
            repo.delete(preset, soft=not hard, created_by='web_ui')
            session.commit()

            return CommandResponse(
                success=True,
                message=f"Preset '{name}' {'permanently deleted' if hard else 'deleted'}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preset: {str(e)}"
        )


# ============================================================================
# Version History
# ============================================================================

@router.get(
    "/presets/{preset_id}/versions",
    response_model=List[Dict[str, Any]],
    summary="Get preset version history",
)
async def get_preset_versions(
    preset_id: int,
    limit: int = Query(50, description="Maximum versions to return"),
) -> List[Dict[str, Any]]:
    """Get version history for a preset."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)
            versions = repo.get_version_history(preset_id, limit=limit)
            return [v.to_dict() for v in versions]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version history: {str(e)}"
        )


@router.post(
    "/presets/{preset_id}/rollback/{version}",
    response_model=Dict[str, Any],
    summary="Rollback preset to version",
)
async def rollback_preset(
    preset_id: int,
    version: int,
) -> Dict[str, Any]:
    """Rollback a preset to a previous version."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import PresetRepository

        with get_session() as session:
            repo = PresetRepository(session)
            preset = repo.rollback_to_version(
                preset_id,
                version,
                created_by='web_ui'
            )
            session.commit()
            return preset.to_dict()

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback: {str(e)}"
        )


# ============================================================================
# Concepts from Database
# ============================================================================

@router.get(
    "/concepts",
    response_model=List[Dict[str, Any]],
    summary="List concepts from database",
)
async def list_db_concepts(
    enabled_only: bool = Query(False, description="Only return enabled concepts"),
) -> List[Dict[str, Any]]:
    """List all concepts from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import ConceptRepository

        with get_session() as session:
            repo = ConceptRepository(session)

            if enabled_only:
                concepts = repo.get_enabled()
            else:
                concepts = repo.get_all()

            return [c.to_dict() for c in concepts]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list concepts: {str(e)}"
        )


@router.post(
    "/concepts",
    response_model=Dict[str, Any],
    summary="Create concept in database",
)
async def create_db_concept(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new concept in database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import ConceptRepository

        with get_session() as session:
            repo = ConceptRepository(session)
            concept = repo.create_from_dict(config, created_by='web_ui')
            session.commit()
            return concept.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create concept: {str(e)}"
        )


@router.put(
    "/concepts/{concept_id}",
    response_model=Dict[str, Any],
    summary="Update concept in database",
)
async def update_db_concept(
    concept_id: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Update a concept in database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import ConceptRepository

        with get_session() as session:
            repo = ConceptRepository(session)
            
            # If config has 'enabled' only, we might want to use toggle_enabled, 
            # but generic update handles it too via update_config
            concept = repo.update_config(
                concept_id,
                config,
                change_description="Updated via UI",
                created_by='web_ui'
            )
            session.commit()
            return concept.to_dict()

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update concept: {str(e)}"
        )


@router.delete(
    "/concepts/{concept_id}",
    response_model=CommandResponse,
    summary="Delete concept from database",
)
async def delete_db_concept(
    concept_id: int,
) -> CommandResponse:
    """Delete a concept from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import ConceptRepository

        with get_session() as session:
            repo = ConceptRepository(session)
            
            concept = repo.get_by_id(concept_id)
            if not concept:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Concept {concept_id} not found"
                )
            
            repo.delete(concept, created_by='web_ui')
            session.commit()
            
            return CommandResponse(
                success=True,
                message=f"Concept {concept_id} deleted"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete concept: {str(e)}"
        )


# ============================================================================
# Samples from Database
# ============================================================================

@router.get(
    "/samples",
    response_model=List[Dict[str, Any]],
    summary="List samples from database",
)
async def list_db_samples(
    enabled_only: bool = Query(False, description="Only return enabled samples"),
) -> List[Dict[str, Any]]:
    """List all samples from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import SampleRepository

        with get_session() as session:
            repo = SampleRepository(session)

            if enabled_only:
                samples = repo.get_enabled()
            else:
                samples = repo.get_all()

            return [s.to_dict() for s in samples]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list samples: {str(e)}"
        )


@router.post(
    "/samples",
    response_model=Dict[str, Any],
    summary="Create sample in database",
)
async def create_db_sample(
    config: Dict[str, Any],
    name: Optional[str] = Query(None, description="Sample name"),
) -> Dict[str, Any]:
    """Create a new sample in database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import SampleRepository

        with get_session() as session:
            repo = SampleRepository(session)
            sample = repo.create_from_dict(config, name=name, created_by='web_ui')
            session.commit()
            return sample.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create sample: {str(e)}"
        )


# ============================================================================
# Training Runs
# ============================================================================

@router.get(
    "/training-runs",
    response_model=List[Dict[str, Any]],
    summary="List training runs",
)
async def list_training_runs(
    limit: int = Query(20, description="Maximum runs to return"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
) -> List[Dict[str, Any]]:
    """List training runs from database."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import TrainingRunRepository

        with get_session() as session:
            repo = TrainingRunRepository(session)

            if status_filter:
                runs = repo.get_by_status(status_filter)
            else:
                runs = repo.get_recent(limit=limit)

            return [r.to_dict() for r in runs]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list training runs: {str(e)}"
        )


@router.get(
    "/training-runs/stats",
    response_model=Dict[str, Any],
    summary="Get training run statistics",
)
async def get_training_stats() -> Dict[str, Any]:
    """Get training run statistics."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.repositories import TrainingRunRepository

        with get_session() as session:
            repo = TrainingRunRepository(session)
            return repo.get_statistics()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ============================================================================
# Export
# ============================================================================

@router.post(
    "/export/presets",
    response_model=Dict[str, Any],
    summary="Export presets to JSON",
)
async def export_presets(
    output_dir: str = Query(..., description="Output directory"),
    include_builtin: bool = Query(True, description="Include builtin presets"),
) -> Dict[str, Any]:
    """Export all presets to JSON files."""
    try:
        from modules.persistence.database import get_session
        from modules.persistence.services.export_service import ExportService

        with get_session() as session:
            service = ExportService(session)
            results = service.export_all_presets(
                Path(output_dir),
                include_builtin=include_builtin
            )
            return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )

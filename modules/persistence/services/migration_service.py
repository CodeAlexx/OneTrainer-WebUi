"""Migration service for JSON to database migration."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from ..models.preset import Preset
from ..models.concept import Concept
from ..models.sample import Sample
from ..repositories.preset_repository import PresetRepository
from ..repositories.concept_repository import ConceptRepository
from ..repositories.sample_repository import SampleRepository


class MigrationService:
    """Service for migrating JSON files to database."""

    def __init__(self, session: Session):
        self.session = session
        self.preset_repo = PresetRepository(session)
        self.concept_repo = ConceptRepository(session)
        self.sample_repo = SampleRepository(session)

    def migrate_presets(
        self,
        presets_dir: Path,
        mark_builtin: bool = True,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Migrate all presets from JSON files.

        Args:
            presets_dir: Path to the presets directory
            mark_builtin: Mark presets starting with '#' as builtin
            created_by: User who initiated the migration

        Returns:
            Dict with migrated, skipped, and error counts
        """
        results = {
            'migrated': [],
            'skipped': [],
            'errors': [],
            'start_time': datetime.utcnow().isoformat()
        }

        if not presets_dir.exists():
            results['error'] = f"Directory not found: {presets_dir}"
            return results

        for json_file in sorted(presets_dir.glob('*.json')):
            try:
                name = json_file.stem

                # Check if already exists
                existing = self.preset_repo.get_by_name(name)
                if existing:
                    results['skipped'].append({
                        'name': name,
                        'reason': 'Already exists'
                    })
                    continue

                # Load JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                # Determine if builtin
                is_builtin = mark_builtin and name.startswith('#')

                # Create preset
                preset = self.preset_repo.create_from_dict(
                    name=name,
                    config_dict=config_dict,
                    is_builtin=is_builtin,
                    created_by=created_by or 'migration'
                )

                results['migrated'].append({
                    'name': name,
                    'id': preset.id,
                    'model_type': preset.model_type
                })

            except json.JSONDecodeError as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': f'JSON parse error: {e}'
                })
            except Exception as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': str(e)
                })

        results['end_time'] = datetime.utcnow().isoformat()
        results['summary'] = {
            'migrated_count': len(results['migrated']),
            'skipped_count': len(results['skipped']),
            'error_count': len(results['errors'])
        }

        return results

    def migrate_concepts(
        self,
        concepts_dir: Path,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Migrate all concepts from JSON files.

        Args:
            concepts_dir: Path to the concepts directory
            created_by: User who initiated the migration

        Returns:
            Dict with migrated, skipped, and error counts
        """
        results = {
            'migrated': [],
            'skipped': [],
            'errors': [],
            'start_time': datetime.utcnow().isoformat()
        }

        if not concepts_dir.exists():
            results['error'] = f"Directory not found: {concepts_dir}"
            return results

        for json_file in sorted(concepts_dir.glob('*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                # Handle both single concept and list of concepts
                concepts_list = content if isinstance(content, list) else [content]

                for concept_dict in concepts_list:
                    name = concept_dict.get('name', json_file.stem)
                    path = concept_dict.get('path', '')

                    # Check if already exists
                    existing = self.concept_repo.get_by_name_and_path(name, path)
                    if existing:
                        results['skipped'].append({
                            'name': name,
                            'path': path,
                            'reason': 'Already exists'
                        })
                        continue

                    # Create concept
                    concept = self.concept_repo.create_from_dict(
                        config_dict=concept_dict,
                        created_by=created_by or 'migration'
                    )

                    results['migrated'].append({
                        'name': name,
                        'id': concept.id,
                        'path': concept.path
                    })

            except json.JSONDecodeError as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': f'JSON parse error: {e}'
                })
            except Exception as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': str(e)
                })

        results['end_time'] = datetime.utcnow().isoformat()
        results['summary'] = {
            'migrated_count': len(results['migrated']),
            'skipped_count': len(results['skipped']),
            'error_count': len(results['errors'])
        }

        return results

    def migrate_samples(
        self,
        samples_dir: Path,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Migrate all samples from JSON files.

        Args:
            samples_dir: Path to the samples directory
            created_by: User who initiated the migration

        Returns:
            Dict with migrated, skipped, and error counts
        """
        results = {
            'migrated': [],
            'skipped': [],
            'errors': [],
            'start_time': datetime.utcnow().isoformat()
        }

        if not samples_dir.exists():
            results['error'] = f"Directory not found: {samples_dir}"
            return results

        for json_file in sorted(samples_dir.glob('*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                # Handle both single sample and list of samples
                samples_list = content if isinstance(content, list) else [content]

                for i, sample_dict in enumerate(samples_list):
                    name = f"{json_file.stem}_{i}" if len(samples_list) > 1 else json_file.stem

                    # Check if already exists by name
                    existing = self.sample_repo.get_by_name(name)
                    if existing:
                        results['skipped'].append({
                            'name': name,
                            'reason': 'Already exists'
                        })
                        continue

                    # Create sample
                    sample = self.sample_repo.create_from_dict(
                        config_dict=sample_dict,
                        name=name,
                        created_by=created_by or 'migration'
                    )

                    results['migrated'].append({
                        'name': name,
                        'id': sample.id,
                        'prompt': sample.prompt[:50] + '...' if len(sample.prompt) > 50 else sample.prompt
                    })

            except json.JSONDecodeError as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': f'JSON parse error: {e}'
                })
            except Exception as e:
                results['errors'].append({
                    'file': str(json_file),
                    'error': str(e)
                })

        results['end_time'] = datetime.utcnow().isoformat()
        results['summary'] = {
            'migrated_count': len(results['migrated']),
            'skipped_count': len(results['skipped']),
            'error_count': len(results['errors'])
        }

        return results

    def migrate_all(
        self,
        onetrainer_root: Path,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run full migration from legacy JSON storage.

        Args:
            onetrainer_root: Path to OneTrainer root directory
            created_by: User who initiated the migration

        Returns:
            Dict with results for each entity type
        """
        results = {
            'presets': self.migrate_presets(
                onetrainer_root / 'training_presets',
                created_by=created_by
            ),
            'concepts': self.migrate_concepts(
                onetrainer_root / 'training_concepts',
                created_by=created_by
            ),
            'samples': self.migrate_samples(
                onetrainer_root / 'training_samples',
                created_by=created_by
            ),
            'migration_time': datetime.utcnow().isoformat()
        }

        # Commit all changes
        self.session.commit()

        return results

    def get_migration_status(self, onetrainer_root: Path) -> Dict[str, Any]:
        """Check migration status - what exists in files vs database.

        Args:
            onetrainer_root: Path to OneTrainer root directory

        Returns:
            Dict with counts of entities in files vs database
        """
        presets_dir = onetrainer_root / 'training_presets'
        concepts_dir = onetrainer_root / 'training_concepts'
        samples_dir = onetrainer_root / 'training_samples'

        # Count JSON files
        preset_files = len(list(presets_dir.glob('*.json'))) if presets_dir.exists() else 0
        concept_files = len(list(concepts_dir.glob('*.json'))) if concepts_dir.exists() else 0
        sample_files = len(list(samples_dir.glob('*.json'))) if samples_dir.exists() else 0

        # Count database records
        db_presets = len(self.preset_repo.get_all())
        db_concepts = len(self.concept_repo.get_all())
        db_samples = len(self.sample_repo.get_all())

        return {
            'files': {
                'presets': preset_files,
                'concepts': concept_files,
                'samples': sample_files
            },
            'database': {
                'presets': db_presets,
                'concepts': db_concepts,
                'samples': db_samples
            },
            'needs_migration': {
                'presets': preset_files > db_presets,
                'concepts': concept_files > db_concepts,
                'samples': sample_files > db_samples
            }
        }

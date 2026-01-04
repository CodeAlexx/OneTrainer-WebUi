"""Export service for database to JSON export."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session

from ..repositories.preset_repository import PresetRepository
from ..repositories.concept_repository import ConceptRepository
from ..repositories.sample_repository import SampleRepository
from ..repositories.training_run_repository import TrainingRunRepository


class ExportService:
    """Service for exporting database entities to JSON."""

    def __init__(self, session: Session):
        self.session = session
        self.preset_repo = PresetRepository(session)
        self.concept_repo = ConceptRepository(session)
        self.sample_repo = SampleRepository(session)
        self.run_repo = TrainingRunRepository(session)

    def export_preset(
        self,
        preset_id: int,
        output_path: Path,
        include_metadata: bool = False
    ) -> Path:
        """Export a single preset to JSON file.

        Args:
            preset_id: ID of the preset to export
            output_path: Directory to save the file
            include_metadata: Include export metadata in JSON

        Returns:
            Path to the created file
        """
        preset = self.preset_repo.get_by_id(preset_id)
        if not preset:
            raise ValueError(f"Preset {preset_id} not found")

        config_dict = preset.get_config_dict()

        if include_metadata:
            config_dict['_export_metadata'] = {
                'exported_from_db': True,
                'original_id': preset.id,
                'original_name': preset.name,
                'export_timestamp': datetime.utcnow().isoformat()
            }

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{preset.name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

        return output_file

    def export_all_presets(
        self,
        output_path: Path,
        include_builtin: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Export all presets to JSON files.

        Args:
            output_path: Directory to save the files
            include_builtin: Include builtin presets
            include_metadata: Include export metadata in JSON

        Returns:
            Dict with export results
        """
        results = {
            'exported': [],
            'errors': [],
            'start_time': datetime.utcnow().isoformat()
        }

        presets = self.preset_repo.get_all()

        for preset in presets:
            if not include_builtin and preset.is_builtin:
                continue

            try:
                file_path = self.export_preset(
                    preset.id,
                    output_path,
                    include_metadata=include_metadata
                )
                results['exported'].append({
                    'id': preset.id,
                    'name': preset.name,
                    'file': str(file_path)
                })
            except Exception as e:
                results['errors'].append({
                    'id': preset.id,
                    'name': preset.name,
                    'error': str(e)
                })

        results['end_time'] = datetime.utcnow().isoformat()
        results['summary'] = {
            'exported_count': len(results['exported']),
            'error_count': len(results['errors'])
        }

        return results

    def export_concepts(
        self,
        concept_ids: List[int],
        output_path: Path,
        filename: str = "concepts.json"
    ) -> Path:
        """Export concepts to a single JSON file.

        Args:
            concept_ids: List of concept IDs to export
            output_path: Directory to save the file
            filename: Name of the output file

        Returns:
            Path to the created file
        """
        concepts = []
        for cid in concept_ids:
            concept = self.concept_repo.get_by_id(cid)
            if concept:
                concepts.append(concept.get_config_dict())

        if not concepts:
            raise ValueError("No valid concepts found to export")

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concepts, f, indent=2)

        return output_file

    def export_all_concepts(
        self,
        output_path: Path,
        filename: str = "concepts.json"
    ) -> Path:
        """Export all concepts to a single JSON file.

        Args:
            output_path: Directory to save the file
            filename: Name of the output file

        Returns:
            Path to the created file
        """
        concepts = self.concept_repo.get_all()
        concept_ids = [c.id for c in concepts]
        return self.export_concepts(concept_ids, output_path, filename)

    def export_samples(
        self,
        sample_ids: List[int],
        output_path: Path,
        filename: str = "samples.json"
    ) -> Path:
        """Export samples to a single JSON file.

        Args:
            sample_ids: List of sample IDs to export
            output_path: Directory to save the file
            filename: Name of the output file

        Returns:
            Path to the created file
        """
        samples = []
        for sid in sample_ids:
            sample = self.sample_repo.get_by_id(sid)
            if sample:
                samples.append(sample.get_config_dict())

        if not samples:
            raise ValueError("No valid samples found to export")

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)

        return output_file

    def export_all_samples(
        self,
        output_path: Path,
        filename: str = "samples.json"
    ) -> Path:
        """Export all samples to a single JSON file.

        Args:
            output_path: Directory to save the file
            filename: Name of the output file

        Returns:
            Path to the created file
        """
        samples = self.sample_repo.get_all()
        sample_ids = [s.id for s in samples]
        return self.export_samples(sample_ids, output_path, filename)

    def export_training_run_package(
        self,
        run_id: int,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export a complete training run with all related entities.

        Args:
            run_id: ID of the training run to export
            output_dir: Directory to save the files

        Returns:
            Dict with paths to created files
        """
        run = self.run_repo.get_by_id(run_id)
        if not run:
            raise ValueError(f"Training run {run_id} not found")

        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}

        # Export config snapshot
        config_file = output_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(run.get_config_snapshot(), f, indent=2)
        files['config'] = config_file

        # Export concepts used
        run_concepts = self.run_repo.get_run_concepts(run_id)
        if run_concepts:
            concepts_data = [rc.get_concept_snapshot() for rc in run_concepts]
            concepts_file = output_dir / "concepts.json"
            with open(concepts_file, 'w', encoding='utf-8') as f:
                json.dump(concepts_data, f, indent=2)
            files['concepts'] = concepts_file

        # Export samples used
        run_samples = self.run_repo.get_run_samples(run_id)
        if run_samples:
            samples_data = [rs.get_sample_snapshot() for rs in run_samples]
            samples_file = output_dir / "samples.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, indent=2)
            files['samples'] = samples_file

        # Export run metadata
        run_metadata = run.to_dict()
        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(run_metadata, f, indent=2)
        files['metadata'] = metadata_file

        return files

    def export_for_sharing(
        self,
        preset_id: int,
        concept_ids: List[int],
        sample_ids: List[int],
        output_dir: Path,
        bundle_name: str = "training_bundle"
    ) -> Path:
        """Create a shareable bundle with preset, concepts, and samples.

        Args:
            preset_id: ID of the preset to include
            concept_ids: List of concept IDs to include
            sample_ids: List of sample IDs to include
            output_dir: Directory to save the bundle
            bundle_name: Name for the bundle directory

        Returns:
            Path to the bundle directory
        """
        bundle_dir = output_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Export preset
        if preset_id:
            self.export_preset(preset_id, bundle_dir)

        # Export concepts
        if concept_ids:
            self.export_concepts(concept_ids, bundle_dir)

        # Export samples
        if sample_ids:
            self.export_samples(sample_ids, bundle_dir)

        # Create bundle manifest
        manifest = {
            'bundle_name': bundle_name,
            'created_at': datetime.utcnow().isoformat(),
            'contents': {
                'preset_id': preset_id,
                'concept_ids': concept_ids,
                'sample_ids': sample_ids
            }
        }

        manifest_file = bundle_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return bundle_dir

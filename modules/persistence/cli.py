"""CLI utilities for database operations."""

import argparse
import sys
from pathlib import Path


def init_db():
    """Initialize the database (create all tables)."""
    from .database import init_database, get_engine
    from .config import DatabaseConfig

    print(f"Initializing database at: {DatabaseConfig.get_db_path()}")
    init_database()
    print("Database initialized successfully!")


def migrate_from_json(onetrainer_root: Path = None, created_by: str = "cli"):
    """Migrate JSON files to database."""
    from .database import get_session
    from .services.migration_service import MigrationService

    if onetrainer_root is None:
        onetrainer_root = Path(__file__).parent.parent.parent

    print(f"Migrating from: {onetrainer_root}")

    with get_session() as session:
        service = MigrationService(session)

        # Check status first
        status = service.get_migration_status(onetrainer_root)
        print("\nCurrent status:")
        print(f"  Files - Presets: {status['files']['presets']}, "
              f"Concepts: {status['files']['concepts']}, "
              f"Samples: {status['files']['samples']}")
        print(f"  DB    - Presets: {status['database']['presets']}, "
              f"Concepts: {status['database']['concepts']}, "
              f"Samples: {status['database']['samples']}")

        # Run migration
        print("\nRunning migration...")
        results = service.migrate_all(onetrainer_root, created_by=created_by)

        # Print results
        for entity_type, result in results.items():
            if entity_type == 'migration_time':
                continue
            print(f"\n{entity_type.upper()}:")
            if 'summary' in result:
                print(f"  Migrated: {result['summary']['migrated_count']}")
                print(f"  Skipped: {result['summary']['skipped_count']}")
                print(f"  Errors: {result['summary']['error_count']}")
                if result['errors']:
                    for err in result['errors']:
                        print(f"    - {err['file']}: {err['error']}")


def export_to_json(output_dir: Path, entity_type: str = "all"):
    """Export database entities to JSON files."""
    from .database import get_session
    from .services.export_service import ExportService

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {output_dir}")

    with get_session() as session:
        service = ExportService(session)

        if entity_type in ("all", "presets"):
            print("\nExporting presets...")
            results = service.export_all_presets(output_dir / "presets")
            print(f"  Exported: {results['summary']['exported_count']}")
            print(f"  Errors: {results['summary']['error_count']}")

        if entity_type in ("all", "concepts"):
            print("\nExporting concepts...")
            try:
                path = service.export_all_concepts(output_dir)
                print(f"  Exported to: {path}")
            except ValueError as e:
                print(f"  No concepts to export: {e}")

        if entity_type in ("all", "samples"):
            print("\nExporting samples...")
            try:
                path = service.export_all_samples(output_dir)
                print(f"  Exported to: {path}")
            except ValueError as e:
                print(f"  No samples to export: {e}")


def show_stats():
    """Show database statistics."""
    from .database import get_session
    from .repositories import PresetRepository, ConceptRepository, SampleRepository, TrainingRunRepository

    with get_session() as session:
        preset_repo = PresetRepository(session)
        concept_repo = ConceptRepository(session)
        sample_repo = SampleRepository(session)
        run_repo = TrainingRunRepository(session)

        print("\nDatabase Statistics:")
        print(f"  Presets: {len(preset_repo.get_all())}")
        print(f"  Concepts: {len(concept_repo.get_all())}")
        print(f"  Samples: {len(sample_repo.get_all())}")

        run_stats = run_repo.get_statistics()
        print(f"  Training Runs: {run_stats['total_runs']}")
        if run_stats['status_counts']:
            print("    By status:")
            for status, count in run_stats['status_counts'].items():
                if count > 0:
                    print(f"      {status}: {count}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="OneTrainer Database CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    subparsers.add_parser("init", help="Initialize the database")

    # migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate JSON files to database")
    migrate_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="OneTrainer root directory"
    )
    migrate_parser.add_argument(
        "--user",
        type=str,
        default="cli",
        help="User name for audit trail"
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for JSON files"
    )
    export_parser.add_argument(
        "--type",
        choices=["all", "presets", "concepts", "samples"],
        default="all",
        help="Entity type to export"
    )

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "migrate":
        migrate_from_json(args.root, args.user)
    elif args.command == "export":
        export_to_json(args.output_dir, args.type)
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

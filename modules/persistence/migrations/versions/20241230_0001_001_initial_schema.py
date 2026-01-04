"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-12-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create presets table
    op.create_table(
        'presets',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('model_type', sa.String(length=100), nullable=False),
        sa.Column('training_method', sa.String(length=100), nullable=False),
        sa.Column('base_model_name', sa.String(length=500), nullable=True),
        sa.Column('peft_type', sa.String(length=50), nullable=True),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('config_version', sa.Integer(), nullable=False, default=10),
        sa.Column('is_builtin', sa.Boolean(), nullable=True, default=False),
        sa.Column('is_favorite', sa.Boolean(), nullable=True, default=False),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('idx_presets_model_type', 'presets', ['model_type'])
    op.create_index('idx_presets_training_method', 'presets', ['training_method'])
    op.create_index('idx_presets_deleted', 'presets', ['deleted_at'])

    # Create concepts table
    op.create_table(
        'concepts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('path', sa.Text(), nullable=False),
        sa.Column('concept_type', sa.String(length=50), nullable=False, default='STANDARD'),
        sa.Column('enabled', sa.Boolean(), nullable=True, default=True),
        sa.Column('image_config_json', sa.Text(), nullable=False, default='{}'),
        sa.Column('text_config_json', sa.Text(), nullable=False, default='{}'),
        sa.Column('concept_stats_json', sa.Text(), nullable=True),
        sa.Column('seed', sa.Integer(), nullable=True),
        sa.Column('include_subdirectories', sa.Boolean(), nullable=True, default=False),
        sa.Column('image_variations', sa.Integer(), nullable=True, default=1),
        sa.Column('text_variations', sa.Integer(), nullable=True, default=1),
        sa.Column('balancing', sa.Float(), nullable=True, default=1.0),
        sa.Column('balancing_strategy', sa.String(length=50), nullable=True, default='REPEATS'),
        sa.Column('loss_weight', sa.Float(), nullable=True, default=1.0),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('config_version', sa.Integer(), nullable=False, default=2),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_concepts_name', 'concepts', ['name'])
    op.create_index('idx_concepts_path', 'concepts', ['path'])
    op.create_index('idx_concepts_type', 'concepts', ['concept_type'])
    op.create_index('idx_concepts_deleted', 'concepts', ['deleted_at'])

    # Create samples table
    op.create_table(
        'samples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('negative_prompt', sa.Text(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=False, default=512),
        sa.Column('height', sa.Integer(), nullable=False, default=512),
        sa.Column('seed', sa.Integer(), nullable=True, default=42),
        sa.Column('random_seed', sa.Boolean(), nullable=True, default=False),
        sa.Column('diffusion_steps', sa.Integer(), nullable=True, default=20),
        sa.Column('cfg_scale', sa.Float(), nullable=True, default=7.0),
        sa.Column('noise_scheduler', sa.String(length=50), nullable=True),
        sa.Column('frames', sa.Integer(), nullable=True, default=1),
        sa.Column('length', sa.Float(), nullable=True, default=10.0),
        sa.Column('config_json', sa.Text(), nullable=False),
        sa.Column('config_version', sa.Integer(), nullable=False, default=0),
        sa.Column('enabled', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_samples_enabled', 'samples', ['enabled'])
    op.create_index('idx_samples_deleted', 'samples', ['deleted_at'])

    # Create training_runs table
    op.create_table(
        'training_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('preset_id', sa.Integer(), nullable=True),
        sa.Column('preset_name', sa.String(length=255), nullable=True),
        sa.Column('run_name', sa.String(length=255), nullable=True),
        sa.Column('workspace_dir', sa.Text(), nullable=True),
        sa.Column('output_model_destination', sa.Text(), nullable=True),
        sa.Column('config_snapshot_json', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, default='pending'),
        sa.Column('current_epoch', sa.Integer(), nullable=True, default=0),
        sa.Column('total_epochs', sa.Integer(), nullable=True),
        sa.Column('current_step', sa.Integer(), nullable=True, default=0),
        sa.Column('total_steps', sa.Integer(), nullable=True),
        sa.Column('metrics_json', sa.Text(), nullable=True),
        sa.Column('final_loss', sa.Float(), nullable=True),
        sa.Column('final_smooth_loss', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('total_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_traceback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['preset_id'], ['presets.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_runs_preset', 'training_runs', ['preset_id'])
    op.create_index('idx_training_runs_status', 'training_runs', ['status'])
    op.create_index('idx_training_runs_started', 'training_runs', ['started_at'])

    # Create training_run_concepts junction table
    op.create_table(
        'training_run_concepts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('concept_id', sa.Integer(), nullable=True),
        sa.Column('concept_snapshot_json', sa.Text(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=True, default=0),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['concept_id'], ['concepts.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_run_concepts_run', 'training_run_concepts', ['training_run_id'])
    op.create_index('idx_run_concepts_concept', 'training_run_concepts', ['concept_id'])

    # Create training_run_samples junction table
    op.create_table(
        'training_run_samples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('sample_id', sa.Integer(), nullable=True),
        sa.Column('sample_snapshot_json', sa.Text(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=True, default=0),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['sample_id'], ['samples.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_run_samples_run', 'training_run_samples', ['training_run_id'])
    op.create_index('idx_run_samples_sample', 'training_run_samples', ['sample_id'])

    # Create generated_samples table
    op.create_table(
        'generated_samples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('global_step', sa.Integer(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=True),
        sa.Column('negative_prompt', sa.Text(), nullable=True),
        sa.Column('seed', sa.Integer(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('diffusion_steps', sa.Integer(), nullable=True),
        sa.Column('cfg_scale', sa.Float(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=True),
        sa.Column('file_format', sa.String(length=20), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('generated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_generated_samples_run', 'generated_samples', ['training_run_id'])
    op.create_index('idx_generated_samples_epoch', 'generated_samples', ['epoch'])
    op.create_index('idx_generated_samples_step', 'generated_samples', ['global_step'])

    # Create entity_versions audit table
    op.create_table(
        'entity_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('data_json', sa.Text(), nullable=False),
        sa.Column('change_type', sa.String(length=20), nullable=False),
        sa.Column('change_description', sa.Text(), nullable=True),
        sa.Column('changed_fields', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('entity_type', 'entity_id', 'version', name='uq_entity_version')
    )
    op.create_index('idx_versions_entity', 'entity_versions', ['entity_type', 'entity_id'])
    op.create_index('idx_versions_created', 'entity_versions', ['created_at'])


def downgrade() -> None:
    op.drop_table('entity_versions')
    op.drop_table('generated_samples')
    op.drop_table('training_run_samples')
    op.drop_table('training_run_concepts')
    op.drop_table('training_runs')
    op.drop_table('samples')
    op.drop_table('concepts')
    op.drop_table('presets')

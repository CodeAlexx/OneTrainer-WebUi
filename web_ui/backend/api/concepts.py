"""
Concepts management REST API endpoints.

Provides endpoints for managing training concepts:
- List concepts from current config
- Add/update/delete concepts
- Sync with training configuration
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, status

from web_ui.backend.models import CommandResponse
from web_ui.backend.services.trainer_service import get_trainer_service

router = APIRouter()


class ConceptData(BaseModel):
    """Concept data for create/update operations."""
    name: str = Field(..., description="Concept name")
    path: str = Field(..., description="Path to concept images")
    enabled: bool = Field(True, description="Whether concept is enabled")
    include_subdirectories: bool = Field(False, description="Include subdirectories")
    image_variations: int = Field(1, description="Image variations per sample")
    text_variations: int = Field(1, description="Text variations per sample")
    balancing: float = Field(1.0, description="Balancing weight")
    balancing_strategy: str = Field("REPEATS", description="Balancing strategy")
    loss_weight: float = Field(1.0, description="Loss weight multiplier")
    seed: int = Field(0, description="Random seed")

    # Prompt settings
    prompt_source: str = Field("FROM_SINGLE_TEXT_FILE", description="Prompt source type")
    prompt_path: Optional[str] = Field(None, description="Path to prompt file")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "my_concept",
                "path": "/path/to/images",
                "enabled": True,
                "include_subdirectories": False,
                "image_variations": 1,
                "text_variations": 1,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "loss_weight": 1.0,
                "seed": 42,
                "prompt_source": "FROM_SINGLE_TEXT_FILE",
                "prompt_path": None,
            }
        }
    }


class ConceptResponse(BaseModel):
    """Response containing concept data."""
    index: int
    concept: ConceptData


class ConceptsListResponse(BaseModel):
    """Response containing list of concepts."""
    concepts: List[ConceptData]
    count: int


def _concept_config_to_data(concept_config) -> ConceptData:
    """Convert a ConceptConfig object to ConceptData."""
    return ConceptData(
        name=getattr(concept_config, "name", ""),
        path=getattr(concept_config, "path", ""),
        enabled=getattr(concept_config, "enabled", True),
        include_subdirectories=getattr(concept_config, "include_subdirectories", False),
        image_variations=getattr(concept_config, "image_variations", 1),
        text_variations=getattr(concept_config, "text_variations", 1),
        balancing=getattr(concept_config, "balancing", 1.0),
        balancing_strategy=str(getattr(concept_config, "balancing_strategy", "REPEATS")),
        loss_weight=getattr(concept_config, "loss_weight", 1.0),
        seed=getattr(concept_config, "seed", 0),
        prompt_source=str(getattr(concept_config, "prompt_source", "FROM_SINGLE_TEXT_FILE")),
        prompt_path=getattr(concept_config, "prompt_path", None),
    )


@router.get(
    "",
    response_model=ConceptsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List concepts",
    description="Get all concepts from current training configuration.",
)
async def list_concepts() -> ConceptsListResponse:
    """
    List all concepts from current config.

    Returns:
        ConceptsListResponse with list of concepts
    """
    trainer_service = get_trainer_service()
    concept_configs = trainer_service.get_concepts()
    
    concepts = [_concept_config_to_data(c) for c in concept_configs]

    return ConceptsListResponse(
        concepts=concepts,
        count=len(concepts)
    )


@router.post(
    "",
    response_model=CommandResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add concept",
    description="Add a new concept to the current configuration.",
)
async def add_concept(concept: ConceptData) -> CommandResponse:
    """
    Add a new concept.

    Args:
        concept: Concept data to add

    Returns:
        CommandResponse indicating success
    """
    trainer_service = get_trainer_service()
    
    success = trainer_service.add_concept(concept.model_dump())
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add concept. Ensure training config is loaded."
        )
    
    return CommandResponse(
        success=True,
        message=f"Concept '{concept.name}' added"
    )


@router.put(
    "/{index}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Update concept",
    description="Update an existing concept by index.",
)
async def update_concept(index: int, concept: ConceptData) -> CommandResponse:
    """
    Update an existing concept.

    Args:
        index: Index of concept to update
        concept: Updated concept data

    Returns:
        CommandResponse indicating success
    """
    trainer_service = get_trainer_service()
    
    success = trainer_service.update_concept(index, concept.model_dump())
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept at index {index} not found or config not loaded"
        )
    
    return CommandResponse(
        success=True,
        message=f"Concept at index {index} updated"
    )


@router.delete(
    "/{index}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete concept",
    description="Remove a concept by index.",
)
async def delete_concept(index: int) -> CommandResponse:
    """
    Delete a concept.

    Args:
        index: Index of concept to delete

    Returns:
        CommandResponse indicating success
    """
    trainer_service = get_trainer_service()
    
    success = trainer_service.delete_concept(index)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept at index {index} not found or config not loaded"
        )
    
    return CommandResponse(
        success=True,
        message=f"Concept at index {index} deleted"
    )

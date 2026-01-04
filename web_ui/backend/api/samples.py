"""
Sample management REST API endpoints.

Provides endpoints for generating and retrieving training samples:
- Generate samples with default or custom parameters
- List and retrieve generated samples
- Access sample images
- Browse sample directory tree
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import re
import os

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web_ui.backend.models import (
    SampleInfo,
    SamplesListResponse,
    SampleGenerateRequest,
    CommandResponse,
)
from web_ui.backend.services.trainer_service import get_trainer_service
from modules.util.config.SampleConfig import SampleConfig

router = APIRouter()


# Default samples directory - typically workspace_dir/samples
DEFAULT_SAMPLES_DIR = Path("./workspace/samples")


class TreeNode(BaseModel):
    """Directory tree node with name, type, and children."""
    name: str
    path: str
    type: str  # "directory" or "prompt" or "image"
    children: Optional[List["TreeNode"]] = None
    image_count: Optional[int] = None


class TreeResponse(BaseModel):
    """Response containing the directory tree."""
    tree: List[TreeNode]
    root_path: str


def _build_samples_tree(root_path: Path) -> List[TreeNode]:
    """
    Build a directory tree for sample browsing.
    
    Handles multiple structures:
    1. Direct: samples_dir/prompt_folder/images
    2. Nested: samples_dir/training_name/samples/prompt_folder/images
    3. Workspace: Look for training workspace dirs in home with samples subdirs
    """
    nodes = []
    
    if not root_path.exists():
        # Try to find samples directories in home
        home = Path.home()
        training_dirs = []
        
        # Look for any directory with a 'samples' subdirectory containing images
        for d in home.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                samples_dir = d / 'samples'
                if samples_dir.exists() and samples_dir.is_dir():
                    # Check if it has prompt subdirs with images
                    has_images = False
                    for subdir in samples_dir.iterdir():
                        if subdir.is_dir():
                            for f in subdir.iterdir():
                                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                                    has_images = True
                                    break
                        if has_images:
                            break
                    if has_images:
                        training_dirs.append(d)
        
        for training_dir in sorted(training_dirs):
            samples_dir = training_dir / 'samples'
            training_node = _build_training_node(training_dir.name, samples_dir)
            if training_node:
                nodes.append(training_node)
        
        return nodes
    
    # Check if root_path itself contains prompt folders (direct structure)
    has_prompt_folders = False
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    
    for item in root_path.iterdir():
        if item.is_dir():
            # Check if this dir contains images (it's a prompt folder)
            images = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
            if images:
                has_prompt_folders = True
                break
    
    if has_prompt_folders:
        # Direct structure: root_path contains prompt folders
        training_node = _build_training_node(root_path.parent.name, root_path)
        if training_node:
            nodes.append(training_node)
    else:
        # Nested structure: look for training dirs inside root_path
        for training_dir in sorted(root_path.iterdir()):
            if not training_dir.is_dir():
                continue
            
            samples_subdir = training_dir / 'samples'
            if samples_subdir.exists():
                training_node = _build_training_node(training_dir.name, samples_subdir)
                if training_node:
                    nodes.append(training_node)
    
    return nodes


def _build_training_node(name: str, samples_dir: Path) -> Optional[TreeNode]:
    """Build a tree node for a training directory's samples."""
    if not samples_dir.exists():
        return None
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    children = []
    
    for prompt_dir in sorted(samples_dir.iterdir()):
        if prompt_dir.is_dir():
            images = [f for f in prompt_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
            if images:
                prompt_node = TreeNode(
                    name=prompt_dir.name,
                    path=str(prompt_dir),
                    type="prompt",
                    image_count=len(images),
                )
                children.append(prompt_node)
    
    if not children:
        return None
    
    return TreeNode(
        name=name,
        path=str(samples_dir),
        type="directory",
        children=children,
    )


@router.get(
    "/tree",
    response_model=TreeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get sample directory tree",
    description="Get a hierarchical tree of sample directories and prompts.",
)
async def get_samples_tree(
    samples_dir: Optional[str] = Query(None, description="Root samples directory path")
) -> TreeResponse:
    """
    Get the directory tree for sample browsing.
    
    Returns:
        TreeResponse with hierarchical tree structure
    """
    root_path = Path(samples_dir) if samples_dir else DEFAULT_SAMPLES_DIR
    tree = _build_samples_tree(root_path)
    
    return TreeResponse(
        tree=tree,
        root_path=str(root_path)
    )


@router.get(
    "/images",
    status_code=status.HTTP_200_OK,
    summary="List images in a directory",
    description="Get a list of images in a specific directory path.",
)
async def list_directory_images(
    path: str = Query(..., description="Directory path to list images from")
) -> Dict[str, Any]:
    """
    List images in a specific directory.
    
    Args:
        path: Directory path
        
    Returns:
        Dict with list of image info
    """
    dir_path = Path(path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {path}"
        )
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    images = []
    
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in image_extensions:
            stat = f.stat()
            images.append({
                "id": f.stem,
                "name": f.name,
                "path": str(f.absolute()),
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    
    return {
        "images": images,
        "count": len(images),
        "directory": str(dir_path)
    }


def _discover_samples(samples_dir: Path, limit: Optional[int] = None) -> List[SampleInfo]:
    """
    Discover sample files in the samples directory.

    Args:
        samples_dir: Directory to search for samples
        limit: Optional limit on number of samples to return

    Returns:
        List of SampleInfo objects
    """
    if not samples_dir.exists():
        return []

    samples = []
    # Common image/video extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.mp4', '*.avi', '*.mov']

    for ext in extensions:
        for sample_file in samples_dir.glob(ext):
            try:
                stat = sample_file.stat()

                # Try to parse metadata from filename
                # Expected format: sample_<step>_<epoch>_<epoch_step>.ext
                match = re.match(r'sample_(\d+)_(\d+)_(\d+)', sample_file.stem)

                sample_info = SampleInfo(
                    id=sample_file.stem,
                    path=str(sample_file.absolute()),
                    filename=sample_file.name,
                    timestamp=datetime.fromtimestamp(stat.st_mtime),
                    epoch=int(match.group(2)) if match else None,
                    step=int(match.group(1)) if match else None,
                    prompt=None,  # Not available from filename
                    seed=None,    # Not available from filename
                )
                samples.append(sample_info)

            except Exception:
                # Skip files that can't be processed
                continue

    # Sort by timestamp (newest first)
    samples.sort(key=lambda s: s.timestamp, reverse=True)

    if limit:
        samples = samples[:limit]

    return samples


@router.get(
    "",
    response_model=SamplesListResponse,
    status_code=status.HTTP_200_OK,
    summary="List generated samples",
    description="Get a list of all generated sample images/videos.",
)
async def list_samples(
    limit: Optional[int] = Query(None, description="Maximum number of samples to return"),
    samples_dir: Optional[str] = Query(None, description="Custom samples directory path")
) -> SamplesListResponse:
    """
    List all generated samples.

    Args:
        limit: Optional limit on number of samples to return
        samples_dir: Optional custom samples directory path

    Returns:
        SamplesListResponse with list of samples
    """
    search_dir = Path(samples_dir) if samples_dir else DEFAULT_SAMPLES_DIR
    samples = _discover_samples(search_dir, limit)

    return SamplesListResponse(
        samples=samples,
        count=len(samples)
    )


@router.post(
    "/generate",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate custom sample",
    description="Trigger generation of a custom sample with specific parameters.",
)
async def generate_sample(request: SampleGenerateRequest) -> CommandResponse:
    """
    Generate a custom sample.

    Args:
        request: Sample generation parameters

    Returns:
        CommandResponse indicating success

    Raises:
        HTTPException: If trainer not initialized or generation fails
    """
    trainer_service = get_trainer_service()

    try:
        # Create SampleConfig from request
        sample_config = SampleConfig.default_values()
        sample_config.prompt = request.prompt
        sample_config.negative_prompt = request.negative_prompt
        sample_config.height = request.height
        sample_config.width = request.width
        sample_config.seed = request.seed
        sample_config.random_seed = request.random_seed
        sample_config.diffusion_steps = request.diffusion_steps
        sample_config.cfg_scale = request.cfg_scale

        # Request custom sample generation
        if not trainer_service.sample_custom(sample_config):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Trainer not initialized or not ready for sampling"
            )

        return CommandResponse(
            success=True,
            message="Custom sample generation requested"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample: {str(e)}"
        )


@router.post(
    "/generate/default",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate default sample",
    description="Trigger generation of a sample using default configuration.",
)
async def generate_default_sample() -> CommandResponse:
    """
    Generate sample using default configuration.

    Returns:
        CommandResponse indicating success

    Raises:
        HTTPException: If trainer not initialized
    """
    trainer_service = get_trainer_service()

    if not trainer_service.sample_default():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trainer not initialized or not ready for sampling"
        )

    return CommandResponse(
        success=True,
        message="Default sample generation requested"
    )


@router.get(
    "/{sample_id}",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get sample file",
    description="Retrieve a specific sample image/video file.",
)
async def get_sample(
    sample_id: str,
    samples_dir: Optional[str] = Query(None, description="Custom samples directory path")
) -> FileResponse:
    """
    Get a sample file by ID.

    Args:
        sample_id: Sample identifier (filename without extension)
        samples_dir: Optional custom samples directory path

    Returns:
        FileResponse with the sample file

    Raises:
        HTTPException: If sample not found
    """
    search_dir = Path(samples_dir) if samples_dir else DEFAULT_SAMPLES_DIR

    if not search_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Samples directory not found: {search_dir}"
        )

    # Search for file with any supported extension
    extensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov']

    for ext in extensions:
        sample_path = search_dir / f"{sample_id}{ext}"
        if sample_path.exists():
            return FileResponse(
                path=str(sample_path),
                media_type=_get_media_type(ext),
                filename=sample_path.name
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Sample '{sample_id}' not found"
    )


def _get_media_type(extension: str) -> str:
    """
    Get MIME type for file extension.

    Args:
        extension: File extension (with leading dot)

    Returns:
        MIME type string
    """
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
    }
    return media_types.get(extension.lower(), 'application/octet-stream')

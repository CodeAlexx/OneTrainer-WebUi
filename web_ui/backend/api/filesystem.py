"""
Filesystem browsing and scanning REST API endpoints.

Provides endpoints for directory browsing and image scanning:
- Browse directory contents (folders + files)
- Scan for images recursively with count and details
- Validate paths for existence and readability
- Serve files directly (for image viewing)
"""

import os
from pathlib import Path
from typing import List, Optional
import mimetypes

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
import imagesize

from web_ui.backend.models import (
    BrowseDirectoryResponse,
    DirectoryEntry,
    ScanDirectoryResponse,
    ImageFileInfo,
    PathValidationResponse,
)

router = APIRouter()

# Common image extensions supported by OneTrainer
SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif', '.gif'}


def is_path_safe(path: Path, allowed_roots: Optional[List[Path]] = None) -> bool:
    """
    Validate that a path is safe to access.

    Prevents directory traversal attacks by ensuring the resolved path
    is under one of the allowed root directories.

    Args:
        path: Path to validate
        allowed_roots: List of allowed root directories (if None, any path is allowed)

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve to absolute path to handle .. and symlinks
        resolved = path.resolve()

        # If no allowed roots specified, allow any existing path
        if allowed_roots is None or len(allowed_roots) == 0:
            return True

        # Check if resolved path is under any allowed root
        for root in allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except ValueError:
                continue

        return False

    except (OSError, RuntimeError):
        return False


@router.get(
    "/browse",
    response_model=BrowseDirectoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Browse directory contents",
    description="List files and folders in a directory with optional filtering by extension.",
)
async def browse_directory(
    path: str = Query(..., description="Directory path to browse"),
    extensions: Optional[str] = Query(
        None,
        description="Comma-separated list of file extensions to filter (e.g., '.jpg,.png')"
    ),
    include_hidden: bool = Query(False, description="Include hidden files and directories"),
) -> BrowseDirectoryResponse:
    """
    Browse directory contents, returning both files and folders.

    Args:
        path: Directory path to browse
        extensions: Optional comma-separated extensions to filter (e.g., ".jpg,.png")
        include_hidden: Whether to include hidden files/directories

    Returns:
        BrowseDirectoryResponse with directory entries

    Raises:
        HTTPException: If path doesn't exist, isn't a directory, or isn't accessible
    """
    try:
        dir_path = Path(path)

        # Validate path exists and is a directory
        if not dir_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path does not exist: {path}"
            )

        if not dir_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {path}"
            )

        # Parse extension filter
        extension_filter = None
        if extensions:
            extension_filter = set(ext.strip().lower() for ext in extensions.split(','))
            # Ensure extensions start with dot
            extension_filter = {ext if ext.startswith('.') else f'.{ext}' for ext in extension_filter}

        # List directory contents
        entries: List[DirectoryEntry] = []

        for item in sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files if not requested
            if not include_hidden and item.name.startswith('.'):
                continue

            is_directory = item.is_dir()

            # Apply extension filter to files
            if not is_directory and extension_filter:
                if item.suffix.lower() not in extension_filter:
                    continue

            # Get file size for files
            size = None
            if not is_directory:
                try:
                    size = item.stat().st_size
                except OSError:
                    pass

            entry = DirectoryEntry(
                name=item.name,
                path=str(item),
                is_directory=is_directory,
                size=size
            )
            entries.append(entry)

        return BrowseDirectoryResponse(
            path=str(dir_path.resolve()),
            entries=entries,
            count=len(entries)
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied accessing path: {path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error browsing directory: {str(e)}"
        )


@router.get(
    "/scan",
    response_model=ScanDirectoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Scan directory for images",
    description="Recursively scan a directory for image files, returning count and file list with optional dimensions.",
)
async def scan_directory_for_images(
    path: str = Query(..., description="Directory path to scan"),
    recursive: bool = Query(True, description="Scan subdirectories recursively"),
    include_dimensions: bool = Query(False, description="Include image dimensions (slower)"),
    extensions: Optional[str] = Query(
        None,
        description="Comma-separated image extensions (e.g., '.jpg,.png'). If not specified, uses default supported types."
    ),
    max_files: Optional[int] = Query(None, description="Maximum number of files to return (for performance)"),
) -> ScanDirectoryResponse:
    """
    Scan directory for image files recursively.

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories
        include_dimensions: Whether to include image dimensions (slower due to file reading)
        extensions: Optional comma-separated extensions filter
        max_files: Optional limit on number of files to return

    Returns:
        ScanDirectoryResponse with image count and file list

    Raises:
        HTTPException: If path doesn't exist, isn't a directory, or isn't accessible
    """
    try:
        dir_path = Path(path)

        # Validate path exists and is a directory
        if not dir_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path does not exist: {path}"
            )

        if not dir_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {path}"
            )

        # Determine which extensions to scan for
        if extensions:
            extension_set = set(ext.strip().lower() for ext in extensions.split(','))
            # Ensure extensions start with dot
            extension_set = {ext if ext.startswith('.') else f'.{ext}' for ext in extension_set}
        else:
            extension_set = SUPPORTED_IMAGE_EXTENSIONS

        # Collect image files
        image_files: List[ImageFileInfo] = []
        total_count = 0

        # Use glob pattern based on recursive flag
        glob_pattern = "**/*" if recursive else "*"

        for item in dir_path.glob(glob_pattern):
            # Only process files with matching extensions
            if not item.is_file():
                continue

            extension = item.suffix.lower()
            if extension not in extension_set:
                continue

            # Skip mask and conditioning label files (OneTrainer convention)
            if item.name.endswith("-masklabel.png") or item.name.endswith("-condlabel.png"):
                continue

            total_count += 1

            # Check if we've hit the max_files limit for detailed info
            if max_files is not None and len(image_files) >= max_files:
                continue

            # Get file info
            try:
                file_size = item.stat().st_size
            except OSError:
                file_size = None

            # Get dimensions if requested
            width = None
            height = None
            if include_dimensions:
                try:
                    # Use imagesize library for fast dimension reading without loading full image
                    w, h = imagesize.get(str(item))
                    if w != -1 and h != -1:  # imagesize returns (-1, -1) for unsupported formats
                        width = w
                        height = h
                except Exception:
                    # Silently skip dimension reading on error
                    pass

            image_info = ImageFileInfo(
                path=str(item),
                filename=item.name,
                size=file_size,
                width=width,
                height=height
            )
            image_files.append(image_info)

        return ScanDirectoryResponse(
            path=str(dir_path.resolve()),
            total_count=total_count,
            files=image_files,
            truncated=max_files is not None and total_count > max_files
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied accessing path: {path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning directory: {str(e)}"
        )


@router.get(
    "/validate-path",
    response_model=PathValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate path",
    description="Check if a path exists and is readable, and determine if it's a file or directory.",
)
async def validate_path(
    path: str = Query(..., description="Path to validate"),
) -> PathValidationResponse:
    """
    Validate that a path exists and is accessible.

    Args:
        path: Path to validate

    Returns:
        PathValidationResponse with validation results
    """
    try:
        check_path = Path(path)

        # Check existence
        exists = check_path.exists()

        if not exists:
            return PathValidationResponse(
                path=path,
                exists=False,
                is_file=False,
                is_directory=False,
                readable=False,
                writable=False,
                message="Path does not exist"
            )

        # Determine type
        is_file = check_path.is_file()
        is_directory = check_path.is_dir()

        # Check permissions
        readable = os.access(check_path, os.R_OK)
        writable = os.access(check_path, os.W_OK)

        # Build message
        type_str = "file" if is_file else "directory" if is_directory else "special file"
        perm_parts = []
        if readable:
            perm_parts.append("readable")
        if writable:
            perm_parts.append("writable")
        perm_str = " and ".join(perm_parts) if perm_parts else "not accessible"

        message = f"Path is a {type_str} and is {perm_str}"

        return PathValidationResponse(
            path=str(check_path.resolve()),
            exists=True,
            is_file=is_file,
            is_directory=is_directory,
            readable=readable,
            writable=writable,
            message=message
        )

    except Exception as e:
        return PathValidationResponse(
            path=path,
            exists=False,
            is_file=False,
            is_directory=False,
            readable=False,
            writable=False,
            message=f"Error validating path: {str(e)}"
        )


@router.get(
    "/file",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Serve file",
    description="Serve a file directly (for image viewing in the browser).",
)
async def serve_file(
    path: str = Query(..., description="Absolute path to the file to serve"),
) -> FileResponse:
    """
    Serve a file directly from the filesystem.
    
    Args:
        path: Absolute path to the file
        
    Returns:
        FileResponse with the file contents
        
    Raises:
        HTTPException: If file doesn't exist or isn't readable
    """
    try:
        file_path = Path(path)
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {path}"
            )
        
        if not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a file: {path}"
            )
        
        # Determine media type
        media_type, _ = mimetypes.guess_type(str(file_path))
        if media_type is None:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
    
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {path}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving file: {str(e)}"
        )


@router.post(
    "/mkdir",
    status_code=status.HTTP_201_CREATED,
    summary="Create directory",
    description="Create a new directory at the specified path.",
)
async def create_directory(
    path: str = Query(..., description="Absolute path for the new directory"),
) -> dict:
    """
    Create a new directory.
    
    Args:
        path: Absolute path for the new directory
        
    Returns:
        Dictionary with success status and created path
        
    Raises:
        HTTPException: If permission denied or other error
    """
    try:
        dir_path = Path(path)
        
        if dir_path.exists():
            if not dir_path.is_dir():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Path exists and is not a directory: {path}"
                )
            return {"success": True, "message": "Directory already exists", "path": str(dir_path)}
        
        # Create directory recursively
        dir_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True, 
            "message": "Directory created successfully", 
            "path": str(dir_path)
        }
    
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied creating directory: {path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating directory: {str(e)}"
        )

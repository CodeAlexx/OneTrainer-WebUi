#!/usr/bin/env python3
"""
OneTrainer Web UI entry point.

This script adds the parent directory to the Python path and starts the
FastAPI application with uvicorn.

Usage:
    python -m web_ui.run          # Production mode (no auto-reload, safe for training)
    python -m web_ui.run --dev    # Development mode (auto-reload on file changes)
"""
import sys
import argparse
from pathlib import Path

# Add the OneTrainer root directory to Python path
# This allows importing from modules/ directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import uvicorn


def main():
    """Run the OneTrainer Web UI server."""
    parser = argparse.ArgumentParser(description='OneTrainer Web UI Server')
    parser.add_argument('--dev', action='store_true',
                        help='Enable development mode with auto-reload (WARNING: stops training on file changes)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()

    mode = "DEVELOPMENT" if args.dev else "PRODUCTION"

    print("=" * 60)
    print("OneTrainer Web UI Server")
    print("=" * 60)
    print(f"Mode: {mode}")
    if args.dev:
        print("  - Auto-reload ENABLED (file changes restart server)")
        print("  - WARNING: Training will stop if backend files change!")
    else:
        print("  - Auto-reload DISABLED (safe for training)")
        print("  - Backend file changes require manual restart")
    print(f"Python path: {sys.path[0]}")
    print(f"Root directory: {root_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    # Run uvicorn server
    uvicorn.run(
        "web_ui.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        reload_dirs=[str(root_dir / "web_ui")] if args.dev else None,
        log_level="info"
    )


if __name__ == "__main__":
    main()

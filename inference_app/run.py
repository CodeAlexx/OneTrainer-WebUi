#!/usr/bin/env python3
"""Run the OneTrainer Inference App."""

import subprocess
import sys
from pathlib import Path

def main():
    app_dir = Path(__file__).parent
    backend_dir = app_dir / "backend"

    # Add to path
    sys.path.insert(0, str(app_dir.parent))

    import uvicorn
    uvicorn.run(
        "inference_app.backend.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )

if __name__ == "__main__":
    main()

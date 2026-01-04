#!/bin/bash
# OneTrainer Web UI Startup Script

set -e

echo "======================================================"
echo "OneTrainer Web UI Server Startup"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "web_ui/run.py" ]; then
    echo "Error: Must run from OneTrainer root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required (found: $python_version)"
    exit 1
fi

echo "✓ Python version: $python_version"

# Check for required packages
echo ""
echo "Checking required packages..."
missing_packages=()

for package in fastapi uvicorn websockets psutil pydantic; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=($package)
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "Error: Missing required packages: ${missing_packages[*]}"
    echo ""
    echo "Install with:"
    echo "  pip install -r web_ui/requirements.txt"
    echo ""
    echo "Or:"
    echo "  pip install fastapi uvicorn[standard] websockets psutil pydantic"
    exit 1
fi

echo "✓ All required packages installed"

# Check for OneTrainer modules
echo ""
echo "Checking OneTrainer modules..."
if ! python3 -c "from modules.util.config.TrainConfig import TrainConfig" 2>/dev/null; then
    echo "Error: Cannot import OneTrainer modules"
    echo "Make sure you're running from the OneTrainer root directory"
    exit 1
fi

echo "✓ OneTrainer modules accessible"

# Test web UI imports
echo ""
echo "Testing web UI imports..."
if ! python3 web_ui/test_imports.py > /dev/null 2>&1; then
    echo "Warning: Import test failed, running verbose test..."
    python3 web_ui/test_imports.py
    exit 1
fi

echo "✓ Web UI imports successful"

# Start the server
echo ""
echo "======================================================"
echo "Starting OneTrainer Web UI Server"
echo "======================================================"
echo ""
echo "Server will be available at:"
echo "  - HTTP: http://localhost:8000"
echo "  - WebSocket: ws://localhost:8000/ws"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================"
echo ""

# Run the server
python3 web_ui/run.py

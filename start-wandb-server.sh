#!/bin/bash
# Start self-hosted WandB server
# Requires Docker to be installed and running

echo "Starting self-hosted WandB server..."
echo "Dashboard will be available at: http://localhost:8080"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if wandb-local container is already running
if docker ps --format '{{.Names}}' | grep -q '^wandb-local$'; then
    echo "WandB server is already running!"
    echo "Dashboard: http://localhost:8080"
    exit 0
fi

# Remove old stopped container if exists
docker rm wandb-local 2>/dev/null

# Start WandB local server
# - Uses Docker volume 'wandb' for persistent storage
# - Exposes port 8080
docker run --rm -d \
    -v wandb:/vol \
    -p 8080:8080 \
    --name wandb-local \
    wandb/local

if [ $? -eq 0 ]; then
    echo ""
    echo "WandB server started successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Open http://localhost:8080 in your browser"
    echo "2. Get a free license from: https://deploy.wandb.ai/"
    echo "3. Paste the license in the /system-admin page"
    echo "4. Create a user account"
    echo "5. Get your API key from Settings > API Keys"
    echo ""
    echo "To use with OneTrainer, set these environment variables:"
    echo "  export WANDB_BASE_URL=http://localhost:8080"
    echo "  export WANDB_API_KEY=your_api_key_here"
    echo ""
    echo "Or run: wandb login --host=http://localhost:8080"
else
    echo "ERROR: Failed to start WandB server"
    exit 1
fi

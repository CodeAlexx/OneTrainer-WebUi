#!/bin/bash
# Stop self-hosted WandB server

echo "Stopping WandB server..."

if docker ps --format '{{.Names}}' | grep -q '^wandb-local$'; then
    docker stop wandb-local
    echo "WandB server stopped."
else
    echo "WandB server is not running."
fi

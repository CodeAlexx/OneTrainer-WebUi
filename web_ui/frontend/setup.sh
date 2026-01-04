#!/bin/bash

echo "Setting up OneTrainer Web UI Frontend..."

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

# Check node version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "Error: Node.js version 18 or higher is required. Current version: $(node -v)"
    exit 1
fi

echo "Node.js version: $(node -v)"
echo "npm version: $(npm -v)"

# Install dependencies
echo "Installing dependencies..."
npm install

# Remove old vite.config.js if it exists
if [ -f "vite.config.js" ]; then
    echo "Removing old vite.config.js..."
    rm vite.config.js
fi

# Remove old .jsx files if they exist
if [ -f "src/main.jsx" ]; then
    echo "Removing old src/main.jsx..."
    rm src/main.jsx
fi

if [ -f "src/App.jsx" ]; then
    echo "Removing old src/App.jsx..."
    rm src/App.jsx
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the development server, run:"
echo "  npm run dev"
echo ""
echo "The app will be available at http://localhost:5173"

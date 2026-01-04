#!/usr/bin/env bash

# OneTrainer Web UI Launch Script
# Starts both the backend (FastAPI) and frontend (Vite) servers
#
# Usage:
#   ./launch-linux.sh          # Production mode (safe for training)
#   ./launch-linux.sh --dev    # Development mode (auto-reload, stops training on changes)

# Ensure correct local path.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
DEV_MODE=false
for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            shift
            ;;
    esac
done

if [ "$DEV_MODE" = true ]; then
    MODE_TEXT="DEVELOPMENT"
    MODE_COLOR="${YELLOW}"
    RELOAD_FLAG="--dev"
else
    MODE_TEXT="PRODUCTION"
    MODE_COLOR="${GREEN}"
    RELOAD_FLAG=""
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║             Trainer Web UI Launcher                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Mode: ${MODE_COLOR}${MODE_TEXT}${NC}"
if [ "$DEV_MODE" = true ]; then
    echo -e "  ${YELLOW}⚠ Auto-reload enabled - training stops on file changes${NC}"
else
    echo -e "  ${GREEN}✓ Safe for training - no auto-reload${NC}"
fi
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    echo -e "Please run: ${YELLOW}python -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Set environment variables
export ONETRAINER_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check if node_modules exists for frontend
if [ ! -d "web_ui/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd web_ui/frontend
    npm install
    cd "$SCRIPT_DIR"
fi

# Kill any existing servers on our ports
echo -e "${YELLOW}Checking for existing servers...${NC}"
pkill -f "web_ui.run" 2>/dev/null
pkill -f "vite.*3000" 2>/dev/null
sleep 1

# Start backend server
echo -e "${GREEN}Starting backend server on port 8000...${NC}"
cd "$SCRIPT_DIR"
if [ "$DEV_MODE" = true ]; then
    nohup python -m web_ui.run --dev > /tmp/onetrainer_backend.log 2>&1 &
else
    nohup python -m web_ui.run > /tmp/onetrainer_backend.log 2>&1 &
fi
BACKEND_PID=$!
echo -e "  Backend PID: ${BLUE}$BACKEND_PID${NC}"

# Wait for backend to start
sleep 2

# Start frontend server
echo -e "${GREEN}Starting frontend server on port 3000...${NC}"
cd "$SCRIPT_DIR/web_ui/frontend"
nohup npm run dev > /tmp/onetrainer_frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "  Frontend PID: ${BLUE}$FRONTEND_PID${NC}"

# Wait for frontend to start
sleep 3

# Get actual frontend port (might be different if 3000 is in use)
FRONTEND_PORT=$(grep -oP 'localhost:\K[0-9]+' /tmp/onetrainer_frontend.log 2>/dev/null | head -1)
FRONTEND_PORT=${FRONTEND_PORT:-3000}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Servers Started!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Frontend:${NC} http://localhost:${FRONTEND_PORT}"
echo -e "  ${CYAN}Backend:${NC}  http://localhost:8000"
echo -e "  ${CYAN}Mode:${NC}     ${MODE_COLOR}${MODE_TEXT}${NC}"
echo ""
echo -e "  ${YELLOW}Logs:${NC}"
echo -e "    Backend:  /tmp/onetrainer_backend.log"
echo -e "    Frontend: /tmp/onetrainer_frontend.log"
echo ""
echo -e "  ${YELLOW}To stop:${NC} ./stop-web-ui.sh"
echo ""

# Optional: Open browser
if command -v xdg-open &> /dev/null; then
    echo -e "${YELLOW}Opening browser...${NC}"
    xdg-open "http://localhost:${FRONTEND_PORT}" 2>/dev/null &
fi

# Keep script running and show logs
echo -e "${YELLOW}Tailing logs (Ctrl+C to stop viewing, servers will continue running)...${NC}"
echo ""
tail -f /tmp/onetrainer_backend.log /tmp/onetrainer_frontend.log

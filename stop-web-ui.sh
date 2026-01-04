#!/usr/bin/env bash

# OneTrainer Web UI Stop Script
# Stops both the backend (FastAPI) and frontend (Vite) servers

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      OneTrainer Web UI Stopper         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Find and kill backend processes
BACKEND_PIDS=$(pgrep -f 'uvicorn.*web_ui' 2>/dev/null)
if [ -n "$BACKEND_PIDS" ]; then
    echo -e "${YELLOW}Stopping backend server(s)...${NC}"
    for PID in $BACKEND_PIDS; do
        echo -e "  Killing PID ${BLUE}$PID${NC}"
        kill $PID 2>/dev/null
    done
else
    echo -e "${YELLOW}No backend server found running${NC}"
fi

# Find and kill frontend processes (Vite)
FRONTEND_PIDS=$(pgrep -f 'vite' 2>/dev/null)
if [ -n "$FRONTEND_PIDS" ]; then
    echo -e "${YELLOW}Stopping frontend server(s)...${NC}"
    for PID in $FRONTEND_PIDS; do
        echo -e "  Killing PID ${BLUE}$PID${NC}"
        kill $PID 2>/dev/null
    done
else
    echo -e "${YELLOW}No frontend server found running${NC}"
fi

# Also kill any node processes related to the web UI (for npm run dev)
NODE_PIDS=$(pgrep -f 'node.*web_ui/frontend' 2>/dev/null)
if [ -n "$NODE_PIDS" ]; then
    echo -e "${YELLOW}Stopping node processes...${NC}"
    for PID in $NODE_PIDS; do
        echo -e "  Killing PID ${BLUE}$PID${NC}"
        kill $PID 2>/dev/null
    done
fi

sleep 1

# Verify they're stopped
REMAINING=$(pgrep -f 'uvicorn.*web_ui' 2>/dev/null)$(pgrep -f 'vite' 2>/dev/null)
if [ -z "$REMAINING" ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         All servers stopped!           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}Some processes may still be running. Force killing...${NC}"
    pkill -9 -f 'uvicorn.*web_ui' 2>/dev/null
    pkill -9 -f 'vite' 2>/dev/null
    sleep 1
    echo -e "${GREEN}Done.${NC}"
fi

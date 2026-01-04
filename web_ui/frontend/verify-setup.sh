#!/bin/bash

echo "=== OneTrainer Frontend Setup Verification ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

# Check Node.js
echo "Checking Node.js..."
node -v > /dev/null 2>&1
check "Node.js installed"

# Check npm
echo "Checking npm..."
npm -v > /dev/null 2>&1
check "npm installed"

# Check TypeScript files
echo ""
echo "Checking TypeScript files..."
[ -f "tsconfig.json" ] && check "tsconfig.json exists"
[ -f "vite.config.ts" ] && check "vite.config.ts exists"
[ -f "src/main.tsx" ] && check "src/main.tsx exists"
[ -f "src/App.tsx" ] && check "src/App.tsx exists"

# Check component files
echo ""
echo "Checking UI components..."
[ -f "src/components/ui/button.tsx" ] && check "Button component"
[ -f "src/components/ui/input.tsx" ] && check "Input component"
[ -f "src/components/ui/card.tsx" ] && check "Card component"
[ -f "src/components/ui/tabs.tsx" ] && check "Tabs component"
[ -f "src/components/ui/select.tsx" ] && check "Select component"
[ -f "src/components/ui/switch.tsx" ] && check "Switch component"
[ -f "src/components/ui/label.tsx" ] && check "Label component"

# Check layout components
echo ""
echo "Checking layout components..."
[ -f "src/components/layout/Header.tsx" ] && check "Header component"
[ -f "src/components/layout/TabNav.tsx" ] && check "TabNav component"
[ -f "src/components/layout/StatusBar.tsx" ] && check "StatusBar component"

# Check lib files
echo ""
echo "Checking lib files..."
[ -f "src/lib/api.ts" ] && check "API client"
[ -f "src/lib/utils.ts" ] && check "Utils"

# Check store
echo ""
echo "Checking state management..."
[ -f "src/stores/configStore.ts" ] && check "Config store"

# Check types
echo ""
echo "Checking types..."
[ -f "src/types/config.ts" ] && check "Type definitions"

# Check if node_modules exists
echo ""
if [ -d "node_modules" ]; then
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} Dependencies not installed. Run: npm install"
fi

echo ""
echo "=== Setup verification complete ==="
echo ""
echo "To start development:"
echo "  1. Install dependencies: npm install"
echo "  2. Start dev server: npm run dev"
echo "  3. Open browser: http://localhost:5173"

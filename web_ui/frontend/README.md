# OneTrainer Web UI Frontend

Modern React + TypeScript frontend for OneTrainer with shadcn/ui components.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library (Radix UI primitives)
- **Zustand** - State management
- **Axios** - API calls
- **Lucide React** - Icons

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn/ui components
│   ├── layout/          # Layout components (Header, StatusBar, etc.)
│   └── tabs/            # Tab content components
├── lib/
│   ├── api.ts           # API client and functions
│   └── utils.ts         # Utility functions (cn helper)
├── stores/
│   └── configStore.ts   # Zustand state management
├── types/
│   └── config.ts        # TypeScript type definitions
├── App.tsx              # Main app component
├── main.tsx             # Entry point
└── index.css            # Global styles

## Features

- Dark theme matching OneTrainer's design
- Cyan/teal accent colors (#22b8cf)
- Tab-based navigation
- Training status bar with progress tracking
- Preset management
- Responsive design
- shadcn/ui component library integration

## API Integration

The frontend connects to the backend API at `http://localhost:8000/api` (proxied in development).

Available API endpoints:
- GET `/api/training/status` - Training status
- POST `/api/training/start` - Start training
- POST `/api/training/stop` - Stop training
- GET `/api/config/presets` - List presets
- GET `/api/config` - Get config
- POST `/api/config` - Save config
- GET `/api/filesystem/browse` - Browse filesystem
- GET `/api/filesystem/scan` - Scan directory

## Color Scheme

- Background: `#1a1a2e`
- Surface: `#252541`
- Border: `#3a3a52`
- Hover: `#2d2d43`
- Primary (accent): `#22b8cf`
- Success: `#10b981`
- Warning: `#f59e0b`
- Danger: `#ef4444`

## Development

The app uses path aliases (`@/`) configured in `tsconfig.json` and `vite.config.ts`.

Example import:
```typescript
import { Button } from '@/components/ui/button';
```

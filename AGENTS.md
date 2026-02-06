# Repository Guidelines

## Project Structure & Module Organization
- `modules/`: core OneTrainer training logic (models, loaders, trainers, utilities).
- `web_ui/`: alpha web interface; `backend/` FastAPI + WebSockets, `frontend/` React/TypeScript (Vite).
- `eritrainer/`: parallel simplified trainer used for EriUI integration.
- `scripts/`: CLI utilities (for example, `train.py`, captioning, conversion tools).
- `configs/`, `training_presets/`, `training_concepts/`: config and preset data.
- `datasets/`, `models/`, `output/`, `workspace/`: data, checkpoints, outputs.
- `docs/` and `resources/`: screenshots and static assets.

## Build, Test, and Development Commands
Backend / CLI:
- `python web_ui/run.py` — start the web UI server.
- `python scripts/train.py --config_path <config.json>` — run OneTrainer CLI training.
- `./start-ui.sh` — convenience launcher that prepares the environment.

Frontend:
- `cd web_ui/frontend && npm install` — install UI dependencies.
- `npm run dev` — Vite dev server.
- `npm run build` — production build.
- `npm run lint` — ESLint checks.

Linting:
- `ruff check .` / `ruff format .` — Python lint + format.

Testing:
- `python web_ui/test_imports.py` — web UI import check.
- `python test_model_loading.py` or `python test_chroma_inference.py` — targeted model tests.
- Additional scripts live in `tests/` and top-level `test_*.py`.

## Coding Style & Naming Conventions
- Python follows Ruff in `pyproject.toml`: 120-char lines, double quotes, and custom import ordering (torch, diffusers, transformers).
- Use Ruff to format before committing; avoid manual formatting tweaks.
- TypeScript/React uses ESLint; follow existing patterns in `web_ui/frontend`.
- Keep naming consistent with existing code (snake_case files, PascalCase classes/components).

## Testing Guidelines
- No coverage target is documented; run the scripts relevant to your changes.
- For model-loading updates, validate with a small local config and a focused test script.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and feature-focused (e.g., "Add ...", "Fix ...", "Update ...").
- PRs should include a concise summary, testing performed, and screenshots for UI changes.
- Link related issues/configs when applicable.

## Safety & Model Loading (EriTrainer)
- Never auto-download models; all `from_pretrained()` calls must use `local_files_only=True`.
- Validate model paths and required files before loading; fail fast if missing.
- Log any file writes or network operations.

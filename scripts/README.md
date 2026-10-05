# Scripts

Utility scripts that are safe to keep under version control.

| Script | Description |
|--------|-------------|
| `start_backend.ps1` | **Quick start** — launches the FastAPI backend on port 8000 |
| `download_checkpoints.py` | Download model checkpoints from GitHub Releases into `backend/checkpoints/` |

## Quick Start

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
```

Generated outputs, caches, local environment files, and large model artifacts should stay out of this directory and are ignored by the repository-level `.gitignore`.

#!/usr/bin/env python3
"""
Main application entry point
Serves both frontend and backend API
Run with: python app.py
"""

import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Create app FIRST - before any imports that might create another app
app = FastAPI(
    title="Face and Body Swap",
    description="Automated face and body swap pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
project_root = Path(__file__).parent.resolve()
templates_path = project_root / "templates"
static_path = project_root / "static"
templates = Jinja2Templates(directory=str(templates_path))

# Mount static files (CSS/JS)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
else:
    print(f"[WARN] Static directory not found at {static_path}")

# ROOT ROUTE - render template (works like Heart-Disease app)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend template"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        }
    )

# Import API router (after app is created to avoid conflicts)
import sys
sys.path.insert(0, str(project_root))
try:
    from src.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1", tags=["API"])
except Exception as e:
    print(f"Warning: Could not load API routes: {e}")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "templates": templates_path.exists(),
        "static": static_path.exists()
    }


def main():
    """Main entry point"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.config import get_config
    
    config = get_config()
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    
    print("=" * 60)
    print("Face and Body Swap - Starting Server")
    print("=" * 60)
    print(f"Templates dir: {templates_path}")
    print(f"Static dir: {static_path}")
    print(f"Server: http://{host}:{port}")
    print("=" * 60)
    print("Visit: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main application entry point
Serves both frontend and backend API
Run with: python app.py
"""

import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import the app from src.api.main (which has startup event for model pre-loading)
import sys
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from src.api.main import app
    # App from main.py already has CORS, routes, and startup event configured
except ImportError as e:
    print(f"ERROR: Could not import app from src.api.main: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Paths
static_path = project_root / "static"
frontend_dist_path = static_path / "dist"

# Ensure required directories exist
required_dirs = ["models", "outputs", "temp", "logs"]
for dir_name in required_dirs:
    dir_path = project_root / dir_name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"WARNING: Could not create directory {dir_name}/: {e}")

# Mount React frontend static assets (JS, CSS, images)
if frontend_dist_path.exists():
    try:
        app.mount("/assets", StaticFiles(directory=str(frontend_dist_path / "assets")), name="assets")
        print(f"[INFO] React frontend assets mounted from: {frontend_dist_path / 'assets'}")
    except Exception as e:
        print(f"[WARN] Could not mount frontend assets: {e}")
else:
    print(f"[WARN] React frontend dist not found at {frontend_dist_path}")
    print(f"[INFO] Run 'cd frontend && npm run build' to build the frontend")

# Mount legacy static files (if any)
if static_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    except Exception as e:
        print(f"[WARN] Could not mount static files: {e}")

# Favicon endpoint - handle browser favicon requests
@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    from fastapi.responses import Response
    return Response(status_code=204)

# ROOT ROUTE - serve React frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend"""
    index_file = frontend_dist_path / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback to old template if React build doesn't exist
        templates_path = project_root / "templates"
        if (templates_path / "index.html").exists():
            from fastapi.templating import Jinja2Templates
            templates = Jinja2Templates(directory=str(templates_path))
            # Create a minimal request object for template rendering
            class MinimalRequest:
                def __init__(self):
                    self.url = None
            return templates.TemplateResponse("index.html", {"request": MinimalRequest()})
        else:
            return HTMLResponse(
                content="""
                <html>
                    <body>
                        <h1>Frontend Not Found</h1>
                        <p>Please build the React frontend:</p>
                        <pre>cd frontend && npm install && npm run build</pre>
                        <p><a href="/docs">API Documentation</a></p>
                    </body>
                </html>
                """,
                status_code=404
            )

# API routes are already included in src.api.main
# No need to import again here

# Health check is already defined in src.api.main
# This endpoint is available at /health


def main():
    """Main entry point"""
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.config import get_config, load_config
    
    # Load config from YAML file if it exists
    config_path = project_root / "configs" / "default.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = get_config()
    
    # Allow environment variable to override port
    port = int(os.getenv("API_PORT", config.get("api", {}).get("port", 8001)))
    host = config.get("api", {}).get("host", "0.0.0.0")
    
    print("=" * 60)
    print("Face and Body Swap - Starting Server")
    print("=" * 60)
    print(f"Frontend dist: {frontend_dist_path}")
    print(f"Static dir: {static_path}")
    print(f"Server: http://{host}:{port}")
    print("=" * 60)
    print("Visit: http://localhost:8000")
    if not frontend_dist_path.exists():
        print("⚠️  React frontend not built. Run: cd frontend && npm run build")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

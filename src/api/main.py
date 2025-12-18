"""FastAPI application main"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from .routes import router
from ..utils.logger import setup_logger, get_logger
from ..utils.config import get_config

# Setup logger
setup_logger()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face and Body Swap API",
    description="Automated face and body swap pipeline API",
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

# Get paths
project_root = Path(__file__).parent.parent.parent.resolve()
frontend_path = project_root / "frontend"
index_file = frontend_path / "index.html"

logger.info(f"Project root: {project_root}")
logger.info(f"Frontend path: {frontend_path}")
logger.info(f"Frontend exists: {frontend_path.exists()}")
logger.info(f"Index.html exists: {index_file.exists()}")

# Root endpoint - MUST be defined BEFORE mounts and routers
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend index page"""
    logger.info("Root route / accessed")
    
    if index_file.exists():
        try:
            # Read and return HTML content
            with open(index_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            logger.info(f"Successfully serving frontend from: {index_file}")
            return HTMLResponse(content=html_content)
        except Exception as e:
            logger.error(f"Error reading index.html: {e}")
            return HTMLResponse(
                content=f"<h1>Error</h1><p>Could not read index.html: {e}</p>",
                status_code=500
            )
    else:
        logger.error(f"Frontend index.html not found at: {index_file}")
        error_html = f"""
        <html>
            <body>
                <h1>Frontend Not Found</h1>
                <p>Expected path: {index_file}</p>
                <p>Frontend path: {frontend_path}</p>
                <p>Project root: {project_root}</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)

# Mount static files (CSS, JS, images) - AFTER root route
if frontend_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
        logger.info(f"✓ Static files mounted at /static from: {frontend_path}")
    except Exception as e:
        logger.error(f"✗ Could not mount static files: {e}")

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["swap"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )

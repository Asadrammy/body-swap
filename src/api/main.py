"""FastAPI application main"""

import time
import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pathlib import Path
from typing import Dict, Any
from .routes import router, jobs
from ..utils.logger import setup_logger, get_logger
from ..utils.config import get_config

# Setup logger with DEBUG level for live logs
setup_logger(log_level="DEBUG")
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

# Add request logging middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info("=" * 80)
        logger.info(f"📥 INCOMING REQUEST: {request.method} {request.url.path}")
        logger.info(f"   Query params: {dict(request.query_params)}")
        logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
        logger.info("=" * 80)
        
        response = await call_next(request)
        
        logger.info(f"📤 RESPONSE: {request.method} {request.url.path} - Status: {response.status_code}")
        
        return response

app.add_middleware(LoggingMiddleware)

# Get paths
project_root = Path(__file__).parent.parent.parent.resolve()
templates_path = project_root / "templates"
static_path = project_root / "static"
index_file = templates_path / "index.html"

logger.info(f"Project root: {project_root}")
logger.info(f"Templates path: {templates_path}")
logger.info(f"Static path: {static_path}")
logger.info(f"Templates exists: {templates_path.exists()}")
logger.info(f"Static exists: {static_path.exists()}")
logger.info(f"Index.html exists: {index_file.exists()}")

# Favicon endpoint - handle browser favicon requests
@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    # Return 204 No Content to suppress 404 errors
    from fastapi.responses import Response
    return Response(status_code=204)

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
                <p>Templates path: {templates_path}</p>
                <p>Project root: {project_root}</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)

# Mount static files (CSS, JS, images) - AFTER root route
if static_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        logger.info(f"✓ Static files mounted at /static from: {static_path}")
    except Exception as e:
        logger.error(f"✗ Could not mount static files: {e}")
else:
    logger.warning(f"Static directory not found at: {static_path}")

# Include API routes
try:
    app.include_router(router, prefix="/api/v1", tags=["swap"])
    logger.info("✓ API router mounted at /api/v1")
    # Log registered routes for debugging
    routes_count = len([r for r in router.routes])
    logger.info(f"✓ Registered {routes_count} API routes")
except Exception as e:
    logger.error(f"✗ Failed to mount API router: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Pre-load models at startup to avoid errors on first request"""
    logger.info("=" * 80)
    logger.info("🚀 PRE-LOADING MODELS AT STARTUP")
    logger.info("=" * 80)
    
    try:
        # Try to load local models (optional - not needed when using Stability AI API)
        from ..models.generator import get_global_generator
        import torch
        
        logger.info("Loading Stable Diffusion models...")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        
        # Get global generator (will load models if not already loaded)
        generator = get_global_generator()
        logger.info(f"Generator device: {generator.device}")
        
        # Verify models are loaded
        if generator.inpaint_pipe is not None:
            logger.info("✅ Models loaded successfully!")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"✅ GPU Memory: {mem:.2f} GB")
                
                # Verify device
                device_str = str(next(generator.inpaint_pipe.unet.parameters()).device)
                logger.info(f"✅ Models on: {device_str}")
            else:
                logger.info("✅ Models loaded on CPU")
        else:
            logger.error("❌ Models failed to load!")
            logger.warning("⚠️  Server will start but refinement may not work until models are downloaded")
            
    except ImportError:
        # get_global_generator not available - this is fine when using Stability AI API
        logger.info("ℹ️  Local models not available - using Stability AI API for refinement (recommended)")
    except Exception as e:
        logger.debug(f"Local model pre-loading skipped: {str(e)}")
        logger.info("ℹ️  Using Stability AI API for refinement (recommended)")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns detailed system health status.
    """
    try:
        # Check GPU availability
        gpu_available = torch.cuda.is_available() if torch is not None else False
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        # Use current drive on Windows, or root on Linux
        import os
        disk_path = os.path.splitdrive(os.getcwd())[0] + '\\' if os.name == 'nt' else '/'
        try:
            disk = psutil.disk_usage(disk_path)
        except:
            # Fallback to current directory
            disk = psutil.disk_usage('.')
        
        # Check active jobs
        active_jobs = sum(1 for job in jobs.values() if job.get("status") in ["pending", "processing"])
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            },
            "gpu": {
                "available": gpu_available,
                "count": gpu_count,
                "devices": [
                    {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                        "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024)
                    } for i in range(gpu_count)
                ] if gpu_available else []
            },
            "api": {
                "active_jobs": active_jobs,
                "total_jobs": len(jobs)
            }
        }
        
        # Determine overall health
        if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
            health_status["status"] = "degraded"
        elif not gpu_available and get_config().get("models", {}).get("generator", {}).get("device") == "cuda":
            health_status["status"] = "warning"  # GPU expected but not available
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-style metrics endpoint for monitoring.
    Returns key performance and system metrics.
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Job metrics
        job_statuses = {}
        for job in jobs.values():
            status = job.get("status", "unknown")
            job_statuses[status] = job_statuses.get(status, 0) + 1
        
        # GPU metrics
        gpu_metrics = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_metrics.append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                    "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
                    "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024),
                    "utilization_percent": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100
                })
        
        metrics = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "percent": disk.percent
                }
            },
            "jobs": {
                "total": len(jobs),
                "by_status": job_statuses,
                "active": sum(1 for job in jobs.values() if job.get("status") in ["pending", "processing"])
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": gpu_metrics
            }
        }
        
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )

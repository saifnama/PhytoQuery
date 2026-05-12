from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from backend.core.http_client import HttpClientManager
from backend.api import ner, ner_pdf, rag, health, doi, search, paper, dashboard
import logging
import os


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the global HTTP client
    await HttpClientManager.get_client()
    logger.info("PhytoQuery backend startup complete.")
    yield
    # Shutdown — order matters here:
    #   1. Close the global HTTP client first (drains in-flight
    #      requests cleanly).
    #   2. Then close the Qdrant local client *before* the Python
    #      interpreter starts tearing down modules. This avoids the
    #      cosmetic "Exception ignored in: QdrantClient.__del__" /
    #      "sys.meta_path is None" traceback that fires when the
    #      destructor runs after the import system is gone. We use
    #      ``peek_rag_service`` (not ``get_rag_service``) so we
    #      never trigger a wasteful lazy-init at shutdown when the
    #      service was never used during this process.
    await HttpClientManager.close_client()
    try:
        from backend.services.rag_engine import peek_rag_service
        svc = peek_rag_service()
        if svc is not None:
            svc.close()
    except Exception as exc:
        logger.warning(f"RAG service shutdown raised (ignored): {exc}")


app = FastAPI(
    title="PhytoQuery Backend",
    description="Production-ready FastAPI backend for NER and RAG on research papers.",
    version="2.0.0",
    lifespan=lifespan,
)


frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "PHYTOQUERY_FRONTEND_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Mount frontend assets (React build)
frontend_dist = os.path.join(os.getcwd(), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_dist, "assets")),
        name="frontend-assets",
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(search.router)
app.include_router(paper.router)
app.include_router(ner.router)
app.include_router(ner_pdf.router)
app.include_router(rag.router)
app.include_router(health.router)
app.include_router(doi.router)
app.include_router(dashboard.router)


# Serve React app for all other routes (SPA routing)
API_PREFIXES = (
    "/search",
    "/paper",
    "/ner",
    "/health",
    "/doi",
    "/api",
    "/static",
    "/assets",
)


@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """Serve the React SPA for all non-API routes."""
    # Don't serve SPA for API routes
    if any(full_path.startswith(prefix.lstrip("/")) for prefix in API_PREFIXES):
        return {"error": "Not found"}

    frontend_dist = os.path.join(os.getcwd(), "frontend", "dist")
    index_path = os.path.join(frontend_dist, "index.html")

    # Check if the path is a file (e.g., favicon.svg)
    file_path = os.path.join(frontend_dist, full_path)
    if full_path and os.path.isfile(file_path):
        return FileResponse(file_path)

    # Otherwise serve index.html for SPA routing
    if os.path.exists(index_path):
        return FileResponse(index_path)

    # Fallback: return 404 if no frontend build exists
    return {"error": "Frontend not built. Run 'cd frontend && bun run build' first."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

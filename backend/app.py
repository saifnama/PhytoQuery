from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from backend.core.http_client import HttpClientManager
from backend.api import ner, rag, health, doi, search, paper
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the global HTTP client
    await HttpClientManager.get_client()
    yield
    # Shutdown: Close the global HTTP client
    await HttpClientManager.close_client()


app = FastAPI(
    title="PhytoQuery Backend",
    description="Production-ready FastAPI backend for NER and RAG on research papers.",
    version="2.0.0",
    lifespan=lifespan,
)

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(search.router)
app.include_router(paper.router)
app.include_router(ner.router)
app.include_router(rag.router)
app.include_router(health.router)
app.include_router(doi.router)


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

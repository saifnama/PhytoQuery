# ─────────────────────────────────────────────────────────────────────────────
# Python 3.14 resource_tracker segfault workaround (must be FIRST)
# ─────────────────────────────────────────────────────────────────────────────
# Python 3.14 rewrote ``multiprocessing.resource_tracker`` to aggressively
# clean up "leaked" POSIX semaphores at interpreter shutdown.  The tracker
# daemon calls ``sem_unlink()`` for every semaphore still in its registry,
# but when the semaphore was already destroyed by the library that created
# it (loky, PyTorch, etc.), the unlink segfaults inside the C extension.
#
# Fix: monkeypatch ``resource_tracker.register`` / ``unregister`` to skip
# semaphore registrations entirely.  Semaphores are still cleaned up by
# their owning libraries' ``__del__`` methods and by our explicit shutdown
# handler (``_close_fastembed_models``) — the tracker's redundant cleanup
# is the one that crashes.  Non-semaphore resources (shared memory, etc.)
# are unaffected.
#
# This block MUST execute before any library creates a multiprocessing
# semaphore (torch, joblib, loky …), so keep it at the very top.
import sys as _sys
if _sys.version_info >= (3, 14):
    try:
        from multiprocessing import resource_tracker as _rt
        _rt__register = _rt.register
        _rt__unregister = _rt.unregister

        def _safe_register(name, rtype):
            if rtype == "semaphore":
                return
            return _rt__register(name, rtype)

        def _safe_unregister(name, rtype):
            if rtype == "semaphore":
                return
            return _rt__unregister(name, rtype)

        _rt.register = _safe_register
        _rt.unregister = _safe_unregister
        del _rt, _rt__register, _rt__unregister
    except Exception:
        pass
del _sys
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Native-extension threading & worker-pool safe defaults
# ─────────────────────────────────────────────────────────────────────────────
# Many ML libraries (fastembed, transformers, torch, numpy via MKL/OpenMP)
# spawn worker processes or thread pools on import. On some Python versions
# (notably 3.14) these pools occasionally crash native code (segfault +
# leaked semaphores) when uvicorn forks workers, or just when many parallel
# tokenization/embedding calls happen at once. The fix is to constrain the
# pool sizes BEFORE the libraries are imported.
#
# Every line below uses ``setdefault`` — so a value you set in your shell
# (``export OMP_NUM_THREADS=8``), ``.env``, or ``.env.<profile>`` ALWAYS
# wins. These are floor-level safe defaults, not opinions.
#
# Defaults chosen for portability across:
#   * any Python version (3.10 → 3.14+)
#   * any machine (laptop, server, HPC)
#   * any library version (fastembed 0.x, torch 2.x, transformers 4.x)
#
# Knobs:
#   JOBLIB_MULTIPROCESSING   When "0", joblib never spawns child processes
#                            via loky — all work runs in the calling thread.
#                            This avoids the POSIX semaphore that triggers
#                            the Python 3.14 resource_tracker segfault.
#   LOKY_MAX_CPU_COUNT       joblib/loky workers (fastembed BM25 uses this).
#                            Setting to 1 limits loky to one child process.
#   TOKENIZERS_PARALLELISM   HuggingFace tokenizers Rust thread pool. "false"
#                            silences the fork-after-parallelism warning and
#                            avoids segfaults when uvicorn forks workers.
#   OMP_NUM_THREADS          OpenMP (torch, numpy). Cap to half CPU count to
#                            leave headroom for FastAPI's event loop +
#                            worker pools sharing the same node.
#   MKL_NUM_THREADS          Intel MKL (numpy on Intel CPUs). Same cap.
#
# Additionally, ``multiprocessing.set_start_method('spawn')`` is forced
# below to prevent a Python 3.14 segfault in the resource_tracker when
# loky's child process is created via fork.  Forked children inherit
# semaphore handles that become invalid during interpreter shutdown;
# spawn avoids this by starting a fresh interpreter in each child.
import os as _os
_os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_cpu_count = _os.cpu_count() or 1
_os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _cpu_count // 2)))
_os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _cpu_count // 2)))
del _cpu_count

import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set
del _mp
# ─────────────────────────────────────────────────────────────────────────────


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

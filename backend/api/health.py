from fastapi import APIRouter, HTTPException
from backend.config import llm_status
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)
@router.get("/ready")
async def readiness_check():
    """
    Check if the service and its dependencies (LLM config, Qdrant) are ready.
    Used by load balancers and orchestrators. The LLM check is
    configuration-only — it never issues a billable model request.
    """
    health_status = {
        "status": "ready",
        "dependencies": {"llm": "unknown", "qdrant": "deferred"},
    }

    # 1. Check LLM configuration (no network call)
    try:
        status = llm_status()
        health_status["dependencies"]["llm"] = status["llm"]
        if status["llm"] != "configured":
            health_status["status"] = "partial"
    except Exception as e:
        logger.error(f"Health check failed for LLM config: {e}")
        health_status["dependencies"]["llm"] = "unreachable"
        health_status["status"] = "partial"

    # 2. Check Qdrant — only if the RAG service has been booted.
    try:
        from backend.services.rag_engine import peek_rag_service
        service = peek_rag_service()
        if service is not None:
            qclient = service._get_qdrant_client()
            qclient.get_collections()
            health_status["dependencies"]["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Health check failed for Qdrant: {e}")
        health_status["dependencies"]["qdrant"] = "down"
        health_status["status"] = "down"

    if health_status["status"] == "down":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

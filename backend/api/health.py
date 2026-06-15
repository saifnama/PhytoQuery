from fastapi import APIRouter, HTTPException
from backend.config import NER_OLLAMA_URL
from backend.core.http_client import HttpClientManager
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)
@router.get("/ready")
async def readiness_check():
    """
    Check if the service and its dependencies (Ollama, Qdrant) are ready.
    Used by load balancers and orchestrators.
    """
    health_status = {
        "status": "ready",
        "dependencies": {"ollama": "unknown", "qdrant": "deferred"},
    }

    # 1. Check Ollama
    try:
        client = await HttpClientManager.get_client()
        # Ollama has a tags endpoint that is fast
        response = await client.get(f"{NER_OLLAMA_URL}/api/tags", timeout=2.0)
        if response.status_code == 200:
            health_status["dependencies"]["ollama"] = "up"
        else:
            health_status["dependencies"]["ollama"] = "down"
            health_status["status"] = "partial"
    except Exception as e:
        logger.error(f"Health check failed for Ollama: {e}")
        health_status["dependencies"]["ollama"] = "unreachable"
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

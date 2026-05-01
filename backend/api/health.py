from fastapi import APIRouter, HTTPException
from backend.config_ner import NER_OLLAMA_URL
from backend.core.http_client import HttpClientManager
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)
@router.get("/ready")
async def readiness_check():
    """
    Check if the service and its dependencies (Ollama, ChromaDB) are ready.
    Used by load balancers and orchestrators.
    """
    health_status = {
        "status": "ready",
        "dependencies": {"ollama": "unknown", "chromadb": "deferred"},
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

    # 2. Check ChromaDB
    try:
        # Simple count check
        from backend.services.rag_engine import peek_rag_service
        service = peek_rag_service()
        if service is not None:
            vectorstore = service._get_user_collection("health_probe")
            vectorstore._collection.count()
            health_status["dependencies"]["chromadb"] = "up"
    except Exception as e:
        logger.error(f"Health check failed for ChromaDB: {e}")
        health_status["dependencies"]["chromadb"] = "down"
        health_status["status"] = "down"

    if health_status["status"] == "down":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

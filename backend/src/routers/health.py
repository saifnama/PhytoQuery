from fastapi import APIRouter, HTTPException
from backend.src.settings import llm_status
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/ready")
async def readiness_check():
    """Readiness for load balancers. Statuses: ready | degraded | down.

    - llm: config-only `llm_status()` plus a cached non-billable `/models`
      dial (`chat/ai/health.py`). Unconfigured/misconfigured keys were
      previously 200/"partial" — now "degraded" so dashboards notice.
    - qdrant: probed only after the RAG service boots; before that it is
      "deferred" (counts as degraded, not ready-green).
    Only "down" (Qdrant error) returns 503.
    """
    health_status = {
        "status": "ready",
        "dependencies": {"llm": "unknown", "qdrant": "deferred"},
    }

    # 1. LLM: config gate, then live (cached) dial.
    try:
        status = llm_status()
        if status["llm"] != "configured":
            health_status["dependencies"]["llm"] = "unconfigured"
            health_status["status"] = "degraded"
        else:
            from backend.src.chat.ai.health import probe_llm

            state, detail = probe_llm()
            health_status["dependencies"]["llm"] = state
            if state != "reachable":
                health_status["status"] = "degraded"
                health_status["dependencies"]["llm_detail"] = detail
    except Exception as e:
        logger.error(f"Health check failed for LLM: {e}")
        health_status["dependencies"]["llm"] = "unreachable"
        health_status["status"] = "degraded"

    # 2. Check Qdrant — only if the RAG service has been booted.
    try:
        from backend.src.chat.service import peek_rag_service
        service = peek_rag_service()
        if service is not None:
            qclient = service._get_qdrant_client()
            qclient.get_collections()
            health_status["dependencies"]["qdrant"] = "up"
        elif health_status["status"] == "ready":
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Health check failed for Qdrant: {e}")
        health_status["dependencies"]["qdrant"] = "down"
        health_status["status"] = "down"

    if health_status["status"] == "down":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

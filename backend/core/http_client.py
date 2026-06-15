import httpx
from typing import Optional

class HttpClientManager:
    """Manages a global httpx.AsyncClient for connection pooling and lifecycle."""
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            # Note: In production, the client should be initialized by the lifespan.
            # This is a fallback to ensure it works even if not explicitly started.
            cls._client = httpx.AsyncClient(timeout=30.0)
        return cls._client

    @classmethod
    async def close_client(cls):
        if cls._client:
            await cls._client.aclose()
            cls._client = None

# Dependency to provide the global client
async def get_http_client() -> httpx.AsyncClient:
    return await HttpClientManager.get_client()

import httpx
from typing import Optional


class HttpClientManager:
    """Manages a global httpx.AsyncClient for connection pooling and lifecycle."""
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None or cls._client.is_closed:
            # Note: In production, the client should be initialized by the lifespan.
            # This is a fallback to ensure it works even if not explicitly started.
            cls._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=5,
                    keepalive_expiry=30,
                ),
            )
        return cls._client

    @classmethod
    async def reset_client(cls):
        """Force-close and discard the current client so the next
        ``get_client()`` call creates a fresh connection pool.  Call
        this when a connection error indicates stale sockets."""
        if cls._client and not cls._client.is_closed:
            try:
                await cls._client.aclose()
            except Exception:
                pass
        cls._client = None

    @classmethod
    async def close_client(cls):
        if cls._client:
            await cls._client.aclose()
            cls._client = None

# Dependency to provide the global client
async def get_http_client() -> httpx.AsyncClient:
    return await HttpClientManager.get_client()

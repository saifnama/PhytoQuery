"""API error contract. Routers raise these; FastAPI renders real status codes.

Replaces the legacy 200-with-{"error": ...} convention (Phase 2): clients can
rely on HTTP status instead of sniffing bodies. Legacy bodies are kept in
`detail` where the frontend's extractErrorDetail already looks.
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    """Base: carries a machine-readable code alongside the message."""

    def __init__(self, status_code: int, message: str, *, code: str = "error"):
        super().__init__(status_code=status_code, detail={"error": message, "code": code})


class NotFoundError(ApiError):
    def __init__(self, message: str = "Not found", *, code: str = "not_found"):
        super().__init__(404, message, code=code)


class UpstreamError(ApiError):
    """Europe PMC / OpenAlex / publisher fetch failed."""

    def __init__(self, message: str = "Upstream source unavailable", *, code: str = "upstream_error"):
        super().__init__(502, message, code=code)


class BadIdentifierError(ApiError):
    def __init__(self, message: str = "Invalid identifier", *, code: str = "bad_identifier"):
        super().__init__(422, message, code=code)


async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


def install_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


def error_body(code: str, message: str, **extra) -> dict:
    # ONE error shape for the whole API → the extension handles errors in one place
    return {"error": {"code": code, "message": message, "request_id": request_id_ctx.get(), **extra}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        # details go to LOGS (for us), only message goes to the CLIENT
        logger.warning(exc.message, extra={"extra_fields": {"code": exc.code, **exc.details}})
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic rejected the request → same error shape instead of FastAPI's default
        return JSONResponse(
            status_code=422,
            content=error_body("invalid_input", "Invalid request", fields=jsonable_encoder(exc.errors())),
        )
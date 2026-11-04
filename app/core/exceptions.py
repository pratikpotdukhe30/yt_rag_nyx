class AppError(Exception):
    """Base for every error WE RAISE ON PURPOSE
    Anything not on APP ERROR = a bug - generic 500, full traced log, nothing leaked to user.
    """
    status_code : int = 500
    code: str = "internal_server_error"

    def __init__(self, message: str, *, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)
        self.message = message

class InvalidInputError(AppError):
    status_code = 400
    code = "invalid_input"

class GuardrailViolationError(AppError):
    status_code = 400
    code = "guardrail_blocked"

class TranscriptNotFoundError(AppError):
    status_code = 404
    code = "transcript_not_found"

class UnsupportedLanguageError(AppError):
    status_code = 422
    code = "unsupported_language"

class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limit_exceeded"

class UpstreamServiceError(AppError):
    status_code = 502
    code = "upstream_server_error"

class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"

class VideoNotIndexedError(AppError):
    status_code = 409          # "conflict with current state": client must ingest first
    code = "video_not_indexed"
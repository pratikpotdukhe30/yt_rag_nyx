import logging
from contextvars import ContextVar
from datetime import timezone, datetime
import json
import sys
from xml.sax import handler

from app.core.config import get_settings

request_id_ctx : ContextVar[str] = ContextVar("request_id", default="-")
class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(), #always utc in logs
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        extra = getattr(record, "extra_fields", None)
        if extra:
            log.update(extra)
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)


def setup_logging() -> None:
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if settings.env == "dev":
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s - %(name)s  [%(request_id)s]- %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)



def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
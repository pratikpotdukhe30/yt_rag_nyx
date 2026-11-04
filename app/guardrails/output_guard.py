import re

from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)

# Matches [mm:ss] and [h:mm:ss]
_TIMESTAMP = re.compile(r"\[(?:(\d+):)?(\d{1,2}):(\d{2})\]")


def cited_seconds(answer: str) -> list[int]:
    return [int(h or 0) * 3600 + int(m) * 60 + int(s) for h, m, s in _TIMESTAMP.findall(answer)]


class OutputGuard:
    # Pure function of (answer, docs): no I/O → trivially unit-testable, ~0 ms
    def is_grounded(self, answer: str, docs: list[Document]) -> bool:
        cited = cited_seconds(answer)
        if not cited:
            return False                       # model was told to cite; no citation = not grounded
        windows = [(d.metadata["start"], d.metadata["end"]) for d in docs]
        # Every citation must land inside a retrieved chunk (±1 s rounding tolerance)
        ok = all(any(start - 1 <= t <= end + 1 for start, end in windows) for t in cited)
        if not ok:
            logger.warning("Ungrounded citation", extra={"extra_fields": {"cited": cited}})
        return ok
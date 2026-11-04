import pytest
from langchain_core.documents import Document

from app.guardrails.output_guard import OutputGuard, cited_seconds

DOCS = [Document(page_content="x", metadata={"start": 3.6, "end": 55.0})]


@pytest.mark.parametrize(   # one test function, many cases → each reported separately
    "answer, expected",
    [
        ("Yes [00:03]", True),          # citation inside the chunk
        ("Yes [00:54]", True),          # near the end, within tolerance
        ("Yes [12:00]", False),         # invented timestamp
        ("Yes, no citation", False),    # missing citation
        ("A [00:10] B [12:00]", False), # one bad citation is enough to fail
    ],
)
def test_is_grounded(answer: str, expected: bool):
    assert OutputGuard().is_grounded(answer, DOCS) is expected


def test_cited_seconds_parses_hours():
    assert cited_seconds("see [1:02:03] and [00:05]") == [3723, 5]
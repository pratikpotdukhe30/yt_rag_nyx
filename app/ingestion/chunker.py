from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.transcript import Transcript

logger = get_logger(__name__)


class TranscriptChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        settings = get_settings()
        # Splitter built ONCE and reused for every transcript → no per-call setup cost.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
            # prefer sentence ends, then commas, then words.
            separators=[". ", "? ", "! ", ", ", " ", ""],
            add_start_index=True,   # LangChain stores each chunk's char offset in metadata["start_index"]
        )

    def split(self, transcript: Transcript) -> list[Document]:
        segments = transcript.segments
        if not segments:
            return []

        # 1. Build the full text AND remember where each segment starts in it.
        #    offsets[i] = char position of segment i → this is our timestamp lookup table.
        texts, offsets, pos = [], [], 0
        for seg in segments:
            clean = seg.text.replace("\n", " ").strip()
            offsets.append(pos)
            texts.append(clean)
            pos += len(clean) + 1                      # +1 = the joining space
        full_text = " ".join(texts)

        # 2. Let LangChain split. Base metadata is copied into every chunk automatically.
        docs = self._splitter.create_documents(
            [full_text],
            metadatas=[{"video_id": transcript.video_id, "language": transcript.language}],
        )

        # 3. Enrich each chunk: char offset → segment → timestamp.
        for idx, doc in enumerate(docs):
            start_char = doc.metadata["start_index"]
            end_char = start_char + len(doc.page_content) - 1

            # bisect = binary search, O(log n). Finds the segment containing that char.
            first = segments[bisect_right(offsets, start_char) - 1]
            last = segments[bisect_right(offsets, end_char) - 1]

            chunk_id = f"{transcript.video_id}:{transcript.language}:{idx}"
            doc.id = chunk_id                          # LangChain-standard id → upsert, no duplicates
            doc.metadata.update(
                chunk_index=idx,
                start=round(first.start, 2),
                end=round(last.start + last.duration, 2),
            )

        logger.info(
            "Transcript chunked",
            extra={"extra_fields": {"video_id": transcript.video_id, "chunks": len(docs)}},
        )
        return docs
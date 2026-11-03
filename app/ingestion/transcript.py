from dataclasses import dataclass
import re
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from app.core.config import get_settings

from app.core.exceptions import (
    InvalidInputError,
    TranscriptNotFoundError,
    UnsupportedLanguageError,
    UpstreamServiceError,
)

from app.core.logging import get_logger
logger = get_logger(__name__)

# YouTube IDs are always 11 chars of [A-Za-z0-9_-]. Validating here = cheap protection
# against garbage/injection reaching a network call.

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# ---------- Our own types: the rest of the app depends on THESE, and not the library ----------

@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

@dataclass(frozen=True)
class Transcript:
    video_id: str
    language: str
    segments: tuple[TranscriptSegment, ...]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)



#The adapter

class TranscriptFetcher:
    def __init__(self, api: YouTubeTranscriptApi | None = None):
        self._api = api or YouTubeTranscriptApi()
        self._supported = get_settings().supported_languages

    def fetch(self, video_id: str, preferred_language: str | None = None) -> Transcript:
        #Validate input before spending a network call
        if not _VIDEO_ID_RE.match(video_id):
            raise InvalidInputError(f"Invalid Youtube video id: details = {video_id}")

        if preferred_language and preferred_language not in self._supported:
            raise UnsupportedLanguageError(f"Unsupported language: {preferred_language} not supported, details: {self._supported}")

        # 2. Language priority: user's choice first, then the other supported ones.
        #    ["it"] requested → tries ["it", "en"]

        languages = [preferred_language] if preferred_language else []
        languages += [lang for lang in self._supported if lang not in languages]

        # 3. Call the vendor, translating ITS errors into OUR domain errors

        try:
            fetched = self._api.fetch(video_id, languages=languages)
        except(TranscriptsDisabled, NoTranscriptFound) as e: # Permanent: retrying won't help → 4xx
            raise TranscriptNotFoundError(
                "No transcript found in supported languages",
                details={"video_id": video_id, "languages": languages},
            ) from e
        except VideoUnavailable as e:
            raise TranscriptNotFoundError("Video unavailable", details={"video_id": video_id}) from e
        except CouldNotRetrieveTranscript as e:
            raise UpstreamServiceError("Failed to retrieve transcript", details={"video_id": video_id}) from e

        # 4. Map vendor objects → our immutable types
        segments = tuple(
            TranscriptSegment(text=s.text, start=s.start, duration=s.duration) for s in fetched

        )

        logger.info(
            "Transcript fetched",
            extra={"extra_fields": {"video_id": video_id, "languages": languages, "segments": len(segments)}},
        )
        return Transcript(video_id= video_id, language= fetched.language_code, segments=segments)



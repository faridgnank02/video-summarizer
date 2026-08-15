"""Time-ordered segment feeds for live rolling summarization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..schemas import Transcript, TranscriptSegment


class SegmentFeed(ABC):
    language: str

    @abstractmethod
    def segments(self) -> AsyncIterator[TranscriptSegment]:
        """Yield transcript segments in chronological order."""
        ...


class WindowedTranscriptFeed(SegmentFeed):
    def __init__(self, transcript: Transcript):
        self.language = transcript.language
        self._transcript = transcript

    async def segments(self) -> AsyncIterator[TranscriptSegment]:
        for seg in self._transcript.segments:
            yield seg

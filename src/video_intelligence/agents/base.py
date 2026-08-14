from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import PipelineContext


class Agent(ABC):
    name: str
    essential: bool = True

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...

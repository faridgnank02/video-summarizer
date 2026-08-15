"""Bounded agentic fact-checking: extract claims, verify each against web search."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models.router import Router, RouterError
from ..schemas import (
    AnalysisReport, Claim, ClaimVerdict, Evidence, FactCheck, PipelineContext,
    QualityPreference, Transcript,
)
from ..search.base import NoSearchProvider, SearchError
from ..search.router import SearchRouter
from .base import Agent
from .prompting import transcript_lines


class ExtractedClaims(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class LoopResponse(BaseModel):
    action: Literal["search", "verdict"]
    query: str | None = None
    verdict: ClaimVerdict | None = None
    confidence: float | None = None
    rationale: str | None = None
    cited_urls: list[str] = Field(default_factory=list)


EXTRACT_PROMPT = """You extract checkable factual claims from a video report.

Return ONLY a JSON object:
{"claims": [{"text": "<a single verifiable factual statement>", "timestamp_s": <float or null>}]}

Rules:
- At most <<MAX_CLAIMS>> claims, each atomic and independently checkable.
- Only objective, verifiable facts. Exclude opinions, predictions, value judgements.
- Set timestamp_s from a [M:SS] marker in the transcript when the claim maps to one, else null.

REPORT SUMMARY:
<<SUMMARY>>

TRANSCRIPT (may be empty):
<<TRANSCRIPT>>"""


VERIFY_PROMPT = """You are fact-checking a single claim against web-search evidence.

CLAIM: <<CLAIM>>

EVIDENCE GATHERED SO FAR:
<<EVIDENCE>>

Return ONLY a JSON object, choosing ONE of:
- To gather more evidence: {"action": "search", "query": "<a better search query>"}
- To decide now: {"action": "verdict", "verdict": "<supported|refuted|misleading|unverified>",
    "confidence": <float 0..1>, "rationale": "<one or two sentences>",
    "cited_urls": ["<url>", ...]}

Verdict meanings:
- supported: evidence corroborates the claim.
- refuted: evidence contradicts the claim.
- misleading: technically true but omits context that changes its meaning.
- unverified: evidence is insufficient to decide.
Only request another search if it is likely to change the verdict."""


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none yet)"
    return "\n".join(f"- {e.title} <{e.url}>: {e.snippet}" for e in evidence)


class FactChecker:
    def __init__(self, router: Router, search: SearchRouter,
                 max_claims: int = 8, max_steps: int = 3, results_per_search: int = 5):
        self._router = router
        self._search = search
        self._max_claims = max_claims
        self._max_steps = max_steps
        self._results_per_search = results_per_search

    async def extract_claims(self, report: AnalysisReport, transcript: Transcript | None,
                             quality: QualityPreference, trace_id: str) -> list[Claim]:
        transcript_text = transcript_lines(transcript)[:12_000] if transcript else ""
        prompt = (EXTRACT_PROMPT
                  .replace("<<MAX_CLAIMS>>", str(self._max_claims))
                  .replace("<<SUMMARY>>", report.summary)
                  .replace("<<TRANSCRIPT>>", transcript_text))
        try:
            result = await self._router.complete(
                task="factcheck", quality=quality, prompt=prompt,
                schema=ExtractedClaims, trace_id=trace_id, stage="factcheck.extract")
        except RouterError:
            return []
        return result.claims[: self._max_claims]

    async def verify_claim(self, claim: Claim, quality: QualityPreference,
                           trace_id: str) -> FactCheck:
        evidence: list[Evidence] = []
        seen: set[str] = set()
        query = claim.text
        steps = 0
        for _ in range(self._max_steps):
            steps += 1
            try:
                results = await self._search.search(query, self._results_per_search)
            except NoSearchProvider:
                raise
            except SearchError:
                return self._unverified(claim, evidence, steps, "search failed")
            for r in results:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    evidence.append(Evidence(title=r.title, url=r.url, snippet=r.snippet))
            prompt = (VERIFY_PROMPT
                      .replace("<<CLAIM>>", claim.text)
                      .replace("<<EVIDENCE>>", _render_evidence(evidence)))
            try:
                resp = await self._router.complete(
                    task="factcheck", quality=quality, prompt=prompt,
                    schema=LoopResponse, trace_id=trace_id, stage="factcheck.verify")
            except RouterError:
                return self._unverified(claim, evidence, steps, "model unavailable")
            if resp.action == "verdict" and resp.verdict is not None:
                cited = [e for e in evidence if e.url in set(resp.cited_urls)] or evidence
                return FactCheck(
                    claim=claim.text, timestamp_s=claim.timestamp_s, verdict=resp.verdict,
                    confidence=resp.confidence, rationale=resp.rationale or "",
                    evidence=cited, search_steps=steps)
            query = resp.query or claim.text
        return self._unverified(claim, evidence, steps,
                                "insufficient evidence within step budget")

    def _unverified(self, claim: Claim, evidence: list[Evidence], steps: int,
                    reason: str) -> FactCheck:
        return FactCheck(claim=claim.text, timestamp_s=claim.timestamp_s,
                         verdict=ClaimVerdict.UNVERIFIED, rationale=reason,
                         evidence=evidence, search_steps=steps)

    async def check(self, claims: list[str], quality: QualityPreference,
                    trace_id: str) -> list[FactCheck]:
        return [await self.verify_claim(Claim(text=c), quality, trace_id) for c in claims]

    async def run(self, report: AnalysisReport, transcript: Transcript | None,
                  quality: QualityPreference, trace_id: str) -> list[FactCheck]:
        claims = await self.extract_claims(report, transcript, quality, trace_id)
        return [await self.verify_claim(c, quality, trace_id) for c in claims]


class FactCheckerAgent(Agent):
    name = "fact_check"
    essential = False

    def __init__(self, checker: FactChecker):
        self._checker = checker

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.options.fact_check:
            return ctx
        if ctx.report is None:
            raise ValueError("fact_check requires a report")
        ctx.report.fact_checks = await self._checker.run(
            ctx.report, ctx.transcript, ctx.options.quality, ctx.trace_id)
        return ctx


def build_search_router(config: dict) -> SearchRouter:
    from ..search.providers.duckduckgo import DuckDuckGoProvider
    from ..search.providers.tavily import TavilyProvider
    api_key_env = config.get("search", {}).get("tavily", {}).get("api_key_env", "TAVILY_API_KEY")
    providers = {
        "tavily": TavilyProvider(api_key_env=api_key_env),
        "duckduckgo": DuckDuckGoProvider(),
    }
    return SearchRouter(config, providers)


def build_factchecker(config_path: str = "config/models.yaml",
                      db_path: str = "data/traces.db") -> FactChecker:
    from ..models.providers.anthropic import AnthropicProvider
    from ..models.providers.ollama import OllamaProvider
    from ..models.providers.openai import OpenAIProvider
    from ..models.router import Router, load_model_config
    from ..tracing import TraceStore

    config = load_model_config(config_path)
    store = TraceStore(db_path)
    router = Router(config, {
        "ollama": OllamaProvider(), "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }, store)
    caps = config.get("fact_check", {})
    return FactChecker(router, build_search_router(config),
                       max_claims=caps.get("max_claims", 8),
                       max_steps=caps.get("max_steps", 3),
                       results_per_search=caps.get("results_per_search", 5))

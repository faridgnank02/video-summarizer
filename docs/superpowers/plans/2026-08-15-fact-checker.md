# Fact-Checker Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, non-essential fact-checking agent that extracts claims from a video report and verifies each against web-search evidence through a bounded agentic loop, exposed both inline in the pipeline and as a sixth MCP tool.

**Architecture:** A new `search/` subpackage mirrors the existing `models/` provider+router pattern (per-vendor client, availability check, first-viable selection, fake for tests). A reusable `FactChecker` service runs the extract→verify loop using the existing model `Router` for reasoning and the new `SearchRouter` for evidence; a thin non-essential `FactCheckerAgent` wraps it for pipeline use, and `mcp_server` drives the same service standalone.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest + pytest-asyncio (`asyncio_mode = auto`), httpx (Tavily), ddgs (DuckDuckGo), FastMCP.

## Global Constraints

- Model IDs and API keys live ONLY in `config/models.yaml` / env vars, never in code.
- Default test suite runs with NO network and NO real Whisper (`addopts = -m "not slow"`); use `FakeProvider` + `FakeSearch`. Real-network tests are marked `@pytest.mark.slow`.
- New dependency floors: `ddgs` (add to `requirements.txt`). Tavily uses the already-present `httpx>=0.27` — no new dep.
- Async tests need no decorator (`asyncio_mode = auto`), but matching the existing `@pytest.mark.asyncio` on MCP tests is fine.
- Test-double location follows the repo convention: shared fakes go in `tests/fakes.py` (where `FakeProvider` already lives), NOT in the `search/` package.
- Verdict vocabulary is exactly `supported | refuted | misleading | unverified`.
- Frozen defaults: `max_claims=8`, `max_steps=3`, `results_per_search=5` (config `fact_check:` block).
- Every model call goes through `Router.complete(task="factcheck", ...)` so it is traced; stages are `factcheck.extract` and `factcheck.verify`.

---

### Task 1: Schemas — Claim, Evidence, ClaimVerdict, FactCheck; report/options fields

**Files:**
- Modify: `src/video_intelligence/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: existing `AnalysisReport`, `JobOptions`, `BaseModel`, `Field`, `StrEnum`.
- Produces:
  - `Claim(text: str, timestamp_s: float | None = None)`
  - `class ClaimVerdict(StrEnum)` → `SUPPORTED="supported"`, `REFUTED="refuted"`, `MISLEADING="misleading"`, `UNVERIFIED="unverified"`
  - `Evidence(title: str, url: str, snippet: str)`
  - `FactCheck(claim: str, timestamp_s: float | None = None, verdict: ClaimVerdict, confidence: float | None = None, rationale: str, evidence: list[Evidence] = [], search_steps: int = 0)`
  - `AnalysisReport.fact_checks: list[FactCheck] = []`
  - `JobOptions.fact_check: bool = False`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, Evidence, FactCheck, JobOptions,
)


def test_factcheck_defaults_and_verdict_enum():
    fc = FactCheck(claim="The sky is blue", verdict=ClaimVerdict.SUPPORTED,
                   rationale="Rayleigh scattering.")
    assert fc.verdict == "supported"
    assert fc.evidence == []
    assert fc.search_steps == 0
    assert fc.timestamp_s is None


def test_claim_and_evidence_shapes():
    c = Claim(text="X happened in 2020")
    assert c.timestamp_s is None
    ev = Evidence(title="T", url="https://e.com", snippet="s")
    assert ev.url == "https://e.com"


def test_report_and_options_gain_factcheck_fields():
    report = AnalysisReport(summary="s", language="en", trace_id="t")
    assert report.fact_checks == []
    assert JobOptions().fact_check is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -k "factcheck or claim_and_evidence or report_and_options" -v`
Expected: FAIL with `ImportError` / `cannot import name 'FactCheck'`.

- [ ] **Step 3: Write minimal implementation**

In `src/video_intelligence/schemas.py`, add the enum after `QualityPreference`:

```python
class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    MISLEADING = "misleading"
    UNVERIFIED = "unverified"
```

Add these models (near `KeyQuote` / before `AnalysisReport`):

```python
class Claim(BaseModel):
    text: str
    timestamp_s: float | None = None


class Evidence(BaseModel):
    title: str
    url: str
    snippet: str


class FactCheck(BaseModel):
    claim: str
    timestamp_s: float | None = None
    verdict: ClaimVerdict
    confidence: float | None = None
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)
    search_steps: int = 0
```

Add to `AnalysisReport`:

```python
    fact_checks: list[FactCheck] = Field(default_factory=list)
```

Add to `JobOptions`:

```python
    fact_check: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (all schema tests, existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/schemas.py tests/test_schemas.py
git commit -m "feat: add fact-check schemas and report/options fields"
```

---

### Task 2: Search subpackage — interface, FakeSearch, SearchRouter

**Files:**
- Create: `src/video_intelligence/search/__init__.py` (empty)
- Create: `src/video_intelligence/search/base.py`
- Create: `src/video_intelligence/search/router.py`
- Modify: `tests/fakes.py` (add `FakeSearch`)
- Modify: `config/models.yaml` (add `search:` section)
- Test: `tests/test_search_router.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `search/base.py`: `SearchResult(title: str, url: str, snippet: str, content: str | None = None)`; `class SearchError(Exception)`; `class NoSearchProvider(SearchError)`; `class SearchProvider(ABC)` with `name: str`, `async def is_available(self) -> bool`, `async def search(self, query: str, k: int) -> list[SearchResult]`.
  - `search/router.py`: `class SearchRouter.__init__(self, config: dict, providers: dict[str, SearchProvider])`; `async def search(self, query: str, k: int) -> list[SearchResult]` (raises `NoSearchProvider` when no candidate is available).
  - `tests/fakes.py`: `class FakeSearch(SearchProvider)` with `__init__(self, name="fakesearch", available=True)`, `enqueue(results: list[SearchResult])`, `.calls: list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_router.py`:

```python
import pytest

from src.video_intelligence.search.base import NoSearchProvider, SearchResult
from src.video_intelligence.search.router import SearchRouter
from tests.fakes import FakeSearch

CONFIG = {"search": {"candidates": ["tavily", "duckduckgo"]}}


async def test_router_picks_first_available():
    tavily = FakeSearch("tavily", available=False)
    ddg = FakeSearch("duckduckgo", available=True)
    ddg.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    router = SearchRouter(CONFIG, {"tavily": tavily, "duckduckgo": ddg})
    results = await router.search("q", k=3)
    assert [r.url for r in results] == ["https://e.com"]
    assert ddg.calls == [{"query": "q", "k": 3}]
    assert tavily.calls == []


async def test_router_raises_when_none_available():
    tavily = FakeSearch("tavily", available=False)
    ddg = FakeSearch("duckduckgo", available=False)
    router = SearchRouter(CONFIG, {"tavily": tavily, "duckduckgo": ddg})
    with pytest.raises(NoSearchProvider):
        await router.search("q", k=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.search'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/search/__init__.py` (empty file).

Create `src/video_intelligence/search/base.py`:

```python
"""Search provider interface: one thin async client per search backend."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None


class SearchError(Exception):
    """A search backend failed for a specific query."""


class NoSearchProvider(SearchError):
    """No configured search backend is available at all."""


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def search(self, query: str, k: int) -> list[SearchResult]: ...
```

Create `src/video_intelligence/search/router.py`:

```python
"""Config-driven first-available selection over search providers."""
from __future__ import annotations

from .base import NoSearchProvider, SearchProvider, SearchResult


class SearchRouter:
    def __init__(self, config: dict, providers: dict[str, SearchProvider]):
        self._candidates = config.get("search", {}).get("candidates", [])
        self._providers = providers

    async def _pick(self) -> SearchProvider:
        for name in self._candidates:
            provider = self._providers.get(name)
            if provider is not None and await provider.is_available():
                return provider
        raise NoSearchProvider("no search provider available")

    async def search(self, query: str, k: int) -> list[SearchResult]:
        provider = await self._pick()
        return await provider.search(query, k)
```

Add to `tests/fakes.py`:

```python
from src.video_intelligence.search.base import SearchProvider, SearchResult


class FakeSearch(SearchProvider):
    def __init__(self, name: str = "fakesearch", available: bool = True):
        self.name = name
        self._available = available
        self._queue: list[list[SearchResult]] = []
        self.calls: list[dict] = []

    def enqueue(self, results: list[SearchResult]) -> None:
        self._queue.append(results)

    async def is_available(self) -> bool:
        return self._available

    async def search(self, query: str, k: int) -> list[SearchResult]:
        self.calls.append({"query": query, "k": k})
        return self._queue.pop(0) if self._queue else []
```

Add to `config/models.yaml` (top level):

```yaml
search:                # ordered candidates; first available wins
  candidates: ["tavily", "duckduckgo"]
  tavily:
    api_key_env: TAVILY_API_KEY   # env var NAME, never a literal key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search_router.py tests/test_config.py -v`
Expected: PASS (router tests pass; `test_config.py` still parses `models.yaml`).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/search tests/fakes.py tests/test_search_router.py config/models.yaml
git commit -m "feat: add search provider interface, SearchRouter, and FakeSearch"
```

---

### Task 3: Real search providers — DuckDuckGo (keyless) and Tavily (httpx)

**Files:**
- Create: `src/video_intelligence/search/providers/__init__.py` (empty)
- Create: `src/video_intelligence/search/providers/duckduckgo.py`
- Create: `src/video_intelligence/search/providers/tavily.py`
- Modify: `requirements.txt` (add `ddgs`)
- Test: `tests/test_search_providers.py`

**Interfaces:**
- Consumes: `SearchProvider`, `SearchResult`, `SearchError` from `search/base.py`.
- Produces:
  - `DuckDuckGoProvider(name="duckduckgo")` — `is_available()` always `True`; `search(query, k)` via `ddgs`.
  - `TavilyProvider(api_key_env: str = "TAVILY_API_KEY")` — `is_available()` true iff env var set; `search(query, k)` via httpx POST.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_providers.py`:

```python
import httpx
import pytest

from src.video_intelligence.search.base import SearchError
from src.video_intelligence.search.providers.tavily import TavilyProvider


async def test_tavily_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider()
    assert await provider.is_available() is False


async def test_tavily_available_with_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    assert await TavilyProvider().is_available() is True


async def test_tavily_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [
            {"title": "A", "url": "https://a.com", "content": "alpha"},
        ]})

    provider = TavilyProvider()
    provider._transport = httpx.MockTransport(handler)  # test seam
    results = await provider.search("q", k=3)
    assert results[0].url == "https://a.com"
    assert results[0].snippet == "alpha"


async def test_tavily_wraps_http_errors(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    provider = TavilyProvider()
    provider._transport = httpx.MockTransport(handler)
    with pytest.raises(SearchError):
        await provider.search("q", k=3)


@pytest.mark.slow
async def test_duckduckgo_real_search():
    from src.video_intelligence.search.providers.duckduckgo import DuckDuckGoProvider
    results = await DuckDuckGoProvider().search("python programming language", k=3)
    assert results and all(r.url for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.search.providers.tavily'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/search/providers/__init__.py` (empty file).

Create `src/video_intelligence/search/providers/tavily.py`:

```python
"""Tavily search via httpx (no vendor SDK)."""
from __future__ import annotations

import os

import httpx

from ..base import SearchError, SearchProvider, SearchResult

_ENDPOINT = "https://api.tavily.com/search"


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key_env: str = "TAVILY_API_KEY"):
        self._api_key = os.environ.get(api_key_env)
        self._transport: httpx.BaseTransport | None = None  # test seam

    async def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, k: int) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=20, transport=self._transport) as client:
                resp = await client.post(_ENDPOINT, json={
                    "api_key": self._api_key, "query": query, "max_results": k,
                })
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise SearchError(f"tavily search failed: {e}") from e
        return [
            SearchResult(title=r.get("title", ""), url=r.get("url", ""),
                         snippet=r.get("content", ""))
            for r in data.get("results", [])
        ]
```

Note: `httpx.AsyncClient(transport=None)` uses the default transport, so production code is unaffected by the test seam.

Create `src/video_intelligence/search/providers/duckduckgo.py`:

```python
"""Keyless DuckDuckGo search via ddgs."""
from __future__ import annotations

import asyncio

from ..base import SearchError, SearchProvider, SearchResult


class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    async def is_available(self) -> bool:
        return True

    async def search(self, query: str, k: int) -> list[SearchResult]:
        try:
            raw = await asyncio.to_thread(self._search_sync, query, k)
        except Exception as e:  # ddgs raises assorted exceptions on failure
            raise SearchError(f"duckduckgo search failed: {e}") from e
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""),
                         snippet=r.get("body", ""))
            for r in raw
        ]

    def _search_sync(self, query: str, k: int) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=k))
```

Add to `requirements.txt` under the ingestion section:

```
# Fact-checking search
ddgs>=6.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search_providers.py -v`
Expected: PASS for the four Tavily tests; the `slow` DuckDuckGo test is deselected by default (`-m "not slow"`).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/search/providers requirements.txt tests/test_search_providers.py
git commit -m "feat: add DuckDuckGo and Tavily search providers"
```

---

### Task 4: FactChecker service — extract, bounded verify loop, check

**Files:**
- Create: `src/video_intelligence/agents/factchecker.py` (service only this task; agent added in Task 5)
- Modify: `config/models.yaml` (add `factcheck` task + `fact_check` caps)
- Test: `tests/test_factchecker.py`

**Interfaces:**
- Consumes: `Router` (`src/video_intelligence/models/router.py`), `SearchRouter` + `NoSearchProvider`/`SearchError` (`search/`), schemas `Claim`, `ClaimVerdict`, `Evidence`, `FactCheck`, `AnalysisReport`, `Transcript`, `QualityPreference`; prompting helper `transcript_lines`.
- Produces:
  - `class ExtractedClaims(BaseModel)`: `claims: list[Claim]`
  - `class LoopResponse(BaseModel)`: `action: Literal["search", "verdict"]`, `query: str | None = None`, `verdict: ClaimVerdict | None = None`, `confidence: float | None = None`, `rationale: str | None = None`, `cited_urls: list[str] = []`
  - `class FactChecker.__init__(self, router: Router, search: SearchRouter, max_claims: int = 8, max_steps: int = 3, results_per_search: int = 5)`
  - `async def extract_claims(self, report: AnalysisReport, transcript: Transcript | None, quality: QualityPreference, trace_id: str) -> list[Claim]`
  - `async def verify_claim(self, claim: Claim, quality: QualityPreference, trace_id: str) -> FactCheck`
  - `async def check(self, claims: list[str], quality: QualityPreference, trace_id: str) -> list[FactCheck]`
  - `async def run(self, report: AnalysisReport, transcript: Transcript | None, quality: QualityPreference, trace_id: str) -> list[FactCheck]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_factchecker.py`:

```python
import pytest

from src.video_intelligence.agents.factchecker import ExtractedClaims, FactChecker, LoopResponse
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, QualityPreference,
)
from src.video_intelligence.search.base import NoSearchProvider, SearchResult
from src.video_intelligence.search.router import SearchRouter
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider, FakeSearch

CONFIG = {
    "tasks": {"factcheck": {"balanced": ["fake/model-x"]}},
    "search": {"candidates": ["fakesearch"]},
}
BAL = QualityPreference.BALANCED


def make(tmp_path, *, available=True):
    fake_model = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake_model}, TraceStore(tmp_path / "t.db"))
    fake_search = FakeSearch("fakesearch", available=available)
    search = SearchRouter(CONFIG, {"fakesearch": fake_search})
    checker = FactChecker(router, search, max_claims=8, max_steps=3, results_per_search=5)
    return checker, fake_model, fake_search


def a_report():
    return AnalysisReport(summary="The Eiffel Tower is 330m tall.",
                          language="en", trace_id="tr1")


async def test_extract_returns_claims_capped(tmp_path):
    checker, model, _ = make(tmp_path)
    checker._max_claims = 2
    model.enqueue(ExtractedClaims(claims=[
        Claim(text="c1"), Claim(text="c2"), Claim(text="c3"),
    ]))
    claims = await checker.extract_claims(a_report(), None, BAL, "tr1")
    assert [c.text for c in claims] == ["c1", "c2"]


async def test_verify_reaches_supported_verdict(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="T", url="https://e.com", snippet="330 metres")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.SUPPORTED,
                               confidence=0.9, rationale="matches source",
                               cited_urls=["https://e.com"]))
    fc = await checker.verify_claim(Claim(text="Eiffel Tower is 330m"), BAL, "tr1")
    assert fc.verdict == "supported"
    assert fc.search_steps == 1
    assert [e.url for e in fc.evidence] == ["https://e.com"]


async def test_verify_refine_triggers_second_search(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="A", url="https://a.com", snippet="vague")])
    search.enqueue([SearchResult(title="B", url="https://b.com", snippet="precise")])
    model.enqueue(LoopResponse(action="search", query="Eiffel Tower height metres"))
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.REFUTED,
                               rationale="source says 300m"))
    fc = await checker.verify_claim(Claim(text="Eiffel Tower is 330m"), BAL, "tr1")
    assert fc.verdict == "refuted"
    assert fc.search_steps == 2
    assert search.calls[1]["query"] == "Eiffel Tower height metres"


async def test_verify_budget_exhausted_is_unverified(tmp_path):
    checker, model, search = make(tmp_path)
    for _ in range(3):
        search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
        model.enqueue(LoopResponse(action="search", query="again"))
    fc = await checker.verify_claim(Claim(text="claim"), BAL, "tr1")
    assert fc.verdict == "unverified"
    assert fc.search_steps == 3


async def test_verify_no_provider_bubbles_up(tmp_path):
    checker, _, _ = make(tmp_path, available=False)
    with pytest.raises(NoSearchProvider):
        await checker.verify_claim(Claim(text="claim"), BAL, "tr1")


async def test_check_wraps_bare_strings(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.MISLEADING,
                               rationale="missing context"))
    out = await checker.check(["some claim"], BAL, "tr1")
    assert len(out) == 1 and out[0].verdict == "misleading"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factchecker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.agents.factchecker'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/agents/factchecker.py`:

```python
"""Bounded agentic fact-checking: extract claims, verify each against web search."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models.router import Router, RouterError
from ..schemas import (
    AnalysisReport, Claim, ClaimVerdict, Evidence, FactCheck, QualityPreference, Transcript,
)
from ..search.base import NoSearchProvider, SearchError
from ..search.router import SearchRouter
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
```

Add to `config/models.yaml` under `tasks:`:

```yaml
  factcheck:
    cheap:    ["ollama/llama3.1:8b", "openai/gpt-4o-mini"]
    balanced: ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]
    best:     ["anthropic/claude-sonnet-5"]
```

Add to `config/models.yaml` (top level):

```yaml
fact_check:            # bounded-loop caps (cost control)
  max_claims: 8
  max_steps: 3
  results_per_search: 5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_factchecker.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/factchecker.py config/models.yaml tests/test_factchecker.py
git commit -m "feat: add FactChecker service with bounded agentic verify loop"
```

---

### Task 5: FactCheckerAgent, pipeline wiring, and FastAPI plumbing

**Files:**
- Modify: `src/video_intelligence/agents/factchecker.py` (add agent + builders)
- Modify: `src/video_intelligence/pipeline.py` (wire agent into `build_pipeline`)
- Modify: `src/api/main.py` (add `fact_check` to the upload form)
- Test: `tests/test_factchecker_agent.py`
- Test: `tests/test_pipeline.py` (extend)

**Interfaces:**
- Consumes: `FactChecker`, `SearchRouter`, `Router`, `TraceStore`, `Agent`, `PipelineContext`, `load_model_config`, provider classes, `DuckDuckGoProvider`, `TavilyProvider`.
- Produces:
  - `class FactCheckerAgent(Agent)`: `name = "fact_check"`, `essential = False`, `__init__(self, checker: FactChecker)`, `async def run(self, ctx) -> ctx`.
  - `def build_search_router(config: dict) -> SearchRouter`
  - `def build_factchecker(config_path: str = "config/models.yaml", db_path: str = "data/traces.db") -> FactChecker`
  - `build_pipeline` now appends a `FactCheckerAgent` as the fifth agent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_factchecker_agent.py`:

```python
import pytest

from src.video_intelligence.agents.factchecker import (
    ExtractedClaims, FactChecker, FactCheckerAgent, LoopResponse,
)
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, JobOptions, PipelineContext, SourceKind,
    VideoSource,
)
from src.video_intelligence.search.base import SearchResult
from src.video_intelligence.search.router import SearchRouter
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider, FakeSearch

CONFIG = {
    "tasks": {"factcheck": {"balanced": ["fake/model-x"]}},
    "search": {"candidates": ["fakesearch"]},
}


def make_agent(tmp_path, *, search_available=True):
    model = FakeProvider("fake")
    router = Router(CONFIG, {"fake": model}, TraceStore(tmp_path / "t.db"))
    search = SearchRouter(CONFIG, {"fakesearch": FakeSearch("fakesearch", available=search_available)})
    checker = FactChecker(router, search)
    return FactCheckerAgent(checker), model, search


def ctx_with_report(fact_check: bool):
    report = AnalysisReport(summary="A claim.", language="en", trace_id="tr1")
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(fact_check=fact_check),
        report=report, trace_id="tr1",
    )


def test_agent_is_non_essential(tmp_path):
    agent, _, _ = make_agent(tmp_path)
    assert agent.essential is False and agent.name == "fact_check"


async def test_agent_noop_when_flag_off(tmp_path):
    agent, model, search = make_agent(tmp_path)
    ctx = await agent.run(ctx_with_report(fact_check=False))
    assert ctx.report.fact_checks == []
    assert model.calls == []


async def test_agent_populates_fact_checks_when_on(tmp_path):
    agent, model, search = make_agent(tmp_path)
    fake_search = next(iter(search._providers.values()))
    model.enqueue(ExtractedClaims(claims=[Claim(text="A claim.")]))
    fake_search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.SUPPORTED,
                               rationale="ok"))
    ctx = await agent.run(ctx_with_report(fact_check=True))
    assert [fc.verdict for fc in ctx.report.fact_checks] == ["supported"]


async def test_agent_raises_when_search_unavailable(tmp_path):
    from src.video_intelligence.search.base import NoSearchProvider
    agent, model, _ = make_agent(tmp_path, search_available=False)
    model.enqueue(ExtractedClaims(claims=[Claim(text="A claim.")]))
    with pytest.raises(NoSearchProvider):
        await agent.run(ctx_with_report(fact_check=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factchecker_agent.py -v`
Expected: FAIL with `ImportError: cannot import name 'FactCheckerAgent'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/video_intelligence/agents/factchecker.py`:

```python
from ..schemas import PipelineContext  # add to existing schema imports
from .base import Agent


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
```

Note: move the top-of-file schema import line to include `PipelineContext` rather than importing twice; the inline `from ..schemas import PipelineContext` above is shown for clarity — fold it into the existing `from ..schemas import (...)` block.

In `src/video_intelligence/pipeline.py`, wire the agent inside `build_pipeline`. After the existing imports block add:

```python
    from .agents.factchecker import FactChecker, FactCheckerAgent, build_search_router
```

Replace the agent list construction so the fifth agent is appended:

```python
    caps = config.get("fact_check", {})
    factchecker = FactChecker(router, build_search_router(config),
                              max_claims=caps.get("max_claims", 8),
                              max_steps=caps.get("max_steps", 3),
                              results_per_search=caps.get("results_per_search", 5))
    return Pipeline(
        [
            Ingestor(workdir=workdir),
            Transcriber(model_name=whisper_model),
            Chapterizer(router),
            Synthesizer(router),
            FactCheckerAgent(factchecker),
        ],
        on_event=on_event,
    )
```

In `src/api/main.py`, add `fact_check` to the upload form. Change the `upload_job` signature and `JobOptions` construction:

```python
    async def upload_job(background: BackgroundTasks,
                         file: UploadFile = File(...),
                         language: str = Form("en"),
                         quality: str = Form("balanced"),
                         force_whisper: bool = Form(False),
                         fact_check: bool = Form(False)) -> dict:
        ...
        options = JobOptions(language=language, quality=QualityPreference(quality),
                             force_whisper=force_whisper, fact_check=fact_check)
```

(The JSON `POST /api/jobs` route already embeds `options: JobOptions`, so `fact_check` flows through it with no change.)

- [ ] **Step 4: Extend the pipeline integration test**

Add to `tests/test_pipeline.py` (uses the file's existing fake-agent style — mirror its existing `FakeAgent`/context helpers; the snippet below is self-contained):

```python
async def test_factchecker_agent_runs_last_and_is_optional():
    from src.video_intelligence.agents.factchecker import FactCheckerAgent
    from src.video_intelligence.pipeline import build_pipeline
    pipeline = build_pipeline.__wrapped__ if hasattr(build_pipeline, "__wrapped__") else None
    # Structural check: the production pipeline lists five agents ending in fact_check.
    import inspect
    src = inspect.getsource(build_pipeline)
    assert "FactCheckerAgent" in src
    assert src.index("Synthesizer") < src.index("FactCheckerAgent")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_factchecker_agent.py tests/test_pipeline.py tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/video_intelligence/agents/factchecker.py src/video_intelligence/pipeline.py src/api/main.py tests/test_factchecker_agent.py tests/test_pipeline.py
git commit -m "feat: wire FactCheckerAgent into pipeline and plumb fact_check through API"
```

---

### Task 6: MCP `fact_check_claims` tool — runtime method + registration

**Files:**
- Modify: `src/mcp_server/runtime.py` (add `fact_check` method + `checker_factory`)
- Modify: `src/mcp_server/server.py` (register the sixth tool)
- Test: `tests/test_mcp_factcheck.py`

**Interfaces:**
- Consumes: `Runtime`, `JobStore`, `build_factchecker`, `AnalysisReport`, `QualityPreference`, `FactChecker.run`/`check`.
- Produces:
  - `Runtime.__init__` gains `checker_factory=build_factchecker`.
  - `async def Runtime.fact_check(self, job_id=None, url=None, claims=None, quality="balanced", language="en", on_event=None) -> dict` — validates exactly one input; returns `{"fact_checks": [...]}` for `job_id`/`claims`, the full report dict for `url`, or `{"status": "failed", "reason": ...}`.
  - `analyze` / `_execute` accept and thread a `fact_check: bool` into `JobOptions`.
  - `server.py` registers `fact_check_claims`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp_factcheck.py`:

```python
import pytest

from src.mcp_server.runtime import Runtime
from src.mcp_server.server import build_server
from src.video_intelligence.schemas import AnalysisReport, FactCheck, ClaimVerdict, StageEvent


class FakeChecker:
    async def run(self, report, transcript, quality, trace_id):
        return [FactCheck(claim="c", verdict=ClaimVerdict.SUPPORTED, rationale="ok")]

    async def check(self, claims, quality, trace_id):
        return [FactCheck(claim=c, verdict=ClaimVerdict.REFUTED, rationale="no") for c in claims]


class FakePipeline:
    def __init__(self, on_event, report):
        self._on_event = on_event
        self._report = report

    async def run(self, source, options):
        return self._report


def make_runtime(tmp_path):
    report = AnalysisReport(summary="s", language="en", trace_id="tr1")

    def pipeline_factory(on_event=None):
        return FakePipeline(on_event, report)

    return Runtime(pipeline_factory=pipeline_factory,
                   checker_factory=lambda **_: FakeChecker(),
                   db_path=tmp_path / "app.db", trace_db=tmp_path / "traces.db")


async def test_fact_check_requires_exactly_one_input(tmp_path):
    rt = make_runtime(tmp_path)
    assert (await rt.fact_check())["status"] == "failed"
    assert (await rt.fact_check(url="u", claims=["a"]))["status"] == "failed"


async def test_fact_check_raw_claims(tmp_path):
    rt = make_runtime(tmp_path)
    out = await rt.fact_check(claims=["the moon is cheese"])
    assert out["fact_checks"][0]["verdict"] == "refuted"


async def test_fact_check_by_job_id_persists(tmp_path):
    rt = make_runtime(tmp_path)
    result = await rt.analyze(url="https://youtu.be/x", async_=False)
    job_id = result["job_id"]
    out = await rt.fact_check(job_id=job_id)
    assert out["fact_checks"][0]["verdict"] == "supported"
    stored = rt.get_report(job_id)
    assert stored["fact_checks"][0]["verdict"] == "supported"


async def test_fact_check_by_url_returns_report(tmp_path):
    rt = make_runtime(tmp_path)
    out = await rt.fact_check(url="https://youtu.be/x")
    assert "summary" in out
    assert out["fact_checks"][0]["verdict"] == "supported"


async def test_server_registers_six_tools(tmp_path):
    report = AnalysisReport(summary="s", language="en", trace_id="tr1")
    rt = Runtime(pipeline_factory=lambda on_event=None: FakePipeline(on_event, report),
                 checker_factory=lambda **_: FakeChecker(),
                 db_path=tmp_path / "a.db", trace_db=tmp_path / "t.db")
    server = build_server(rt)
    names = {t.name for t in await server.list_tools()}
    assert "fact_check_claims" in names and len(names) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_factcheck.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'checker_factory'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mcp_server/runtime.py`:

Add import at top:

```python
from src.video_intelligence.agents.factchecker import build_factchecker
from src.video_intelligence.schemas import AnalysisReport
```

Extend `__init__`:

```python
    def __init__(self, pipeline_factory=build_pipeline,
                 checker_factory=build_factchecker,
                 db_path: str | Path = "data/app.db",
                 trace_db: str | Path = "data/traces.db"):
        self.factory = pipeline_factory
        self.checker_factory = checker_factory
        self.jobs = JobStore(db_path)
        self.traces = TraceStore(trace_db)
```

Thread `fact_check` through `analyze` (add param + into `JobOptions`):

```python
    async def analyze(self, url: str, quality: str = "balanced",
                      language: str = "en", force_whisper: bool = False,
                      fact_check: bool = False, async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
        try:
            quality_pref = QualityPreference(quality)
        except ValueError:
            return {"status": "failed", "reason": f"invalid quality: {quality!r}"}
        source = VideoSource(kind=SourceKind.YOUTUBE, url=url)
        options = JobOptions(language=language, quality=quality_pref,
                             force_whisper=force_whisper, fact_check=fact_check)
        job_id = uuid.uuid4().hex
        self.jobs.create(job_id, source, options)
        if async_:
            task = asyncio.create_task(self._execute(job_id, source, options, _noop))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return {"status": "running", "job_id": job_id}
        return await self._execute(job_id, source, options, on_event or _noop)
```

Add the `fact_check` method:

```python
    async def fact_check(self, job_id: str | None = None, url: str | None = None,
                         claims: list[str] | None = None, quality: str = "balanced",
                         language: str = "en",
                         on_event: EventCallback | None = None) -> dict:
        provided = [x is not None for x in (job_id, url, claims)]
        if sum(provided) != 1:
            return {"status": "failed",
                    "reason": "provide exactly one of job_id, url, or claims"}
        try:
            quality_pref = QualityPreference(quality)
        except ValueError:
            return {"status": "failed", "reason": f"invalid quality: {quality!r}"}

        if url is not None:
            return await self.analyze(url=url, quality=quality, language=language,
                                      fact_check=True, async_=False,
                                      on_event=on_event)

        checker = self.checker_factory()
        if claims is not None:
            import uuid as _uuid
            results = await checker.check(claims, quality_pref, _uuid.uuid4().hex)
            return {"fact_checks": [fc.model_dump() for fc in results]}

        job = self.jobs.get(job_id)
        if job is None or job["report"] is None:
            return {"status": "failed", "reason": "no completed report for job_id"}
        report = AnalysisReport.model_validate(job["report"])
        report.fact_checks = await checker.run(report, None, quality_pref, report.trace_id)
        self.jobs.update(job_id, report=report)
        return {"fact_checks": [fc.model_dump() for fc in report.fact_checks]}
```

In `src/mcp_server/server.py`, register the tool (add before `return mcp`):

```python
    @mcp.tool()
    async def fact_check_claims(ctx: Context, job_id: str | None = None,
                                url: str | None = None,
                                claims: list[str] | None = None,
                                quality: QualityPreference = QualityPreference.BALANCED,
                                language: str = "en") -> dict:
        """Fact-check a video's claims. Provide exactly one of job_id, url, or claims."""
        return await runtime.fact_check(
            job_id=job_id, url=url, claims=claims, quality=quality,
            language=language, on_event=_progress(ctx))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp_factcheck.py tests/test_mcp_runtime.py tests/test_mcp_server.py -v`
Expected: PASS. Note: `test_mcp_server.py::test_server_registers_all_five_tools` asserts an exact set of five names — update its expected set to include `fact_check_claims` (six names) as part of this step.

- [ ] **Step 5: Update the existing five-tool assertion**

In `tests/test_mcp_server.py`, change the expected names set:

```python
    assert names == {"analyze_video", "get_job_status", "get_report",
                     "extract_chapters", "get_trace", "fact_check_claims"}
```

- [ ] **Step 6: Commit**

```bash
git add src/mcp_server/runtime.py src/mcp_server/server.py tests/test_mcp_factcheck.py tests/test_mcp_server.py
git commit -m "feat: add fact_check_claims MCP tool (job_id, url, or raw claims)"
```

---

### Task 7: Full-suite verification and dependency install

**Files:** none (verification only).

- [ ] **Step 1: Install the new dependency**

Run: `pip install -r requirements.txt`
Expected: `ddgs` installs; no other changes.

- [ ] **Step 2: Run the full default suite**

Run: `pytest`
Expected: PASS — the prior 78 tests plus the new fact-checker/search/MCP tests, all with no network (slow tests deselected).

- [ ] **Step 3: Run the slow suite (network-gated, optional)**

Run: `pytest -m slow -k duckduckgo`
Expected: PASS when network is available (real DuckDuckGo search returns results); acceptable to skip offline.

- [ ] **Step 4: Commit any lockfile/doc touch-ups if needed**

```bash
git status   # expect clean if steps 1-3 changed nothing tracked
```

---

## Self-Review

**Spec coverage:**
- Non-essential opt-in agent after Synthesizer → Task 5 (`essential=False`, no-op when flag off, degrades on `NoSearchProvider`). ✓
- Reusable `FactChecker` service backing both agent and MCP tool → Tasks 4 (service), 5 (agent), 6 (MCP). ✓
- `search/` subpackage mirroring `models/` (interface, router, two providers, fake) → Tasks 2–3. ✓
- Config-driven first-available search selection → Task 2 (`SearchRouter`). ✓
- Bounded agentic loop (extract, per-claim refine up to `max_steps`, budget→unverified) → Task 4. ✓
- Verdict vocabulary `supported|refuted|misleading|unverified` → Task 1 enum + Task 4 prompt. ✓
- Schemas `Claim`/`Evidence`/`FactCheck`, `AnalysisReport.fact_checks`, `JobOptions.fact_check` → Task 1. ✓
- `fact_check_claims` MCP tool with `job_id`/`url`/`claims`, mutual-exclusivity error, job_id persists → Task 6. ✓
- Config additions (`factcheck` task, `fact_check` caps, `search` section) → Tasks 2 & 4. ✓
- FastAPI plumbing (flag through `POST /jobs` + upload form), no new route → Task 5. ✓
- Frontend deferred → not in plan (matches spec "Out"). ✓
- Tests: FactChecker paths, SearchRouter fallback, agent no-op/degrade, three MCP modes + validation, slow keyless smoke → Tasks 2–7. ✓
- Deps: `ddgs` added, Tavily via existing httpx → Task 3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `FactChecker(router, search, max_claims, max_steps, results_per_search)`, `verify_claim(claim, quality, trace_id)`, `run(report, transcript, quality, trace_id)`, `check(claims, quality, trace_id)`, `SearchRouter(config, providers).search(query, k)`, `SearchResult(title,url,snippet,content?)`, `FactCheck(...)` fields — all consistent across Tasks 1, 4, 5, 6. `NoSearchProvider` (bubbles/degrades) vs `SearchError` (per-claim → unverified) distinguished consistently in base, router, providers, and service. ✓

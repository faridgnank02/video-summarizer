# Fact-Checker Agent — Design

**Date:** 2026-08-15
**Status:** Approved (design), pending spec review
**Phase:** 4 of the "Agentic Multimodal Video Intelligence" transformation
**Branch:** `claude/fact-checker-phase-4` (off `claude/mcp-server-phase-2`)

## Goal

Add claim verification to the pipeline: a fifth, **non-essential** agent
(`fact_check`) that runs after the Synthesizer, extracts checkable claims from
the report, and verifies each one against web-search evidence through a bounded
agentic loop, attaching per-claim verdicts with citations. The verdict logic is
factored into a reusable `FactChecker` service so the same code backs the
`fact_check_claims` MCP tool that phase 2 reserved but left absent.

This is the first phase to change the `video_intelligence` core since phase 1
(new `search/` subpackage, new schemas, an extra agent). That is expected: the
fact-checker is genuinely new intelligence, not a new adapter.

## Scope

**In:**

- A `search/` subpackage mirroring the `models/` provider+router pattern: a
  `SearchProvider` interface, two real providers (Tavily, DuckDuckGo), a
  `FakeSearch` for the network-free test suite, and a config-driven
  `SearchRouter` that picks the first available provider.
- A `FactChecker` service implementing the bounded agentic loop, and a thin
  non-essential `FactCheckerAgent` wrapping it for opt-in inline use.
- New schemas: `Evidence`, `FactCheck`, `ClaimVerdict`; `AnalysisReport` gains
  `fact_checks`; `JobOptions` gains `fact_check`.
- A sixth MCP tool, `fact_check_claims`, accepting `job_id` **or** `url` **or**
  `claims`.
- Config: a `factcheck` task + caps in `models.yaml`, and a `search:` section
  listing ordered candidate providers.
- FastAPI: plumb the `fact_check` flag through the existing `POST /jobs` request
  model (no new route).
- Tests with `FakeSearch` + the existing `FakeProvider`; one slow keyless smoke.

**Out (deferred):**

- **All frontend work.** No submit-form toggle, no fact-check rendering in the
  report view this phase. The backend produces `fact_checks`; surfacing it in
  the SPA is a later, separate slice.
- A dedicated FastAPI fact-check route (the MCP tool covers standalone use;
  a REST route is YAGNI).
- Visual agent, live-stream mode, deployment re-fit, evaluation re-wire (their
  own phases, unchanged).

## Architecture

```
src/video_intelligence/
  search/                    # NEW — mirrors models/ (provider interface + router)
    __init__.py
    base.py                  # SearchProvider ABC, SearchResult, SearchError
    router.py                # SearchRouter: first-available selection from config
    providers/
      __init__.py
      tavily.py              # httpx call to Tavily; key from env named in config
      duckduckgo.py          # keyless, via ddgs
    fake.py                  # FakeSearch — canned/scripted results for tests
  agents/
    factchecker.py           # FactCheckerAgent (non-essential) + FactChecker service
  schemas.py                 # + Evidence, FactCheck, ClaimVerdict;
                             #   AnalysisReport.fact_checks, JobOptions.fact_check
```

Key decisions:

- **Non-essential agent, degrade-friendly.** `FactCheckerAgent.essential =
  False`, `name = "fact_check"`. It runs after the Synthesizer, so `ctx.report`
  already exists. If the `fact_check` option is off it is a **no-op** (returns
  `ctx` unchanged — not a degraded stage). If it is on and verification fails
  (e.g. no search provider available), the pipeline's existing policy appends
  `fact_check` to `degraded_stages` and the report still ships.
- **Reusable service, thin agent.** All logic lives in a `FactChecker` service
  (`extract_claims`, `verify_claim`, `check`). The agent wraps it for inline
  use; `mcp_server/runtime.py` calls the same service for the standalone tool.
  Inline and standalone verification therefore cannot drift.
- **Search mirrors models.** The `SearchProvider` interface and `SearchRouter`
  copy the `models/` pattern (per-vendor thin client, availability check at call
  time, first viable candidate from config, fake for tests). Adding a search
  backend is one file, exactly like adding a model provider.
- **Reuse the model Router for reasoning.** Claim extraction and verdict
  reasoning are `router.complete(task="factcheck", ...)` calls, so they are
  cost/latency-traced through the existing `TraceStore` like every other model
  call, and tier-selectable (`cheap|balanced|best`).

## The Bounded Agentic Loop

`FactChecker` owns three methods:

1. **`extract_claims(report, transcript, max_claims) -> list[Claim]`** (`Claim`
   = `{text, timestamp_s?}`, see schemas) — one
   cheap-tier LLM call over `report.summary` (and, when short enough, the
   transcript) returning up to `max_claims` atomic, checkable factual claims.
   Each claim carries an optional `timestamp_s` located from the transcript's
   `[M:SS]` markers when the claim maps to a specific moment. Opinions,
   predictions, and value judgements are instructed out.

2. **`verify_claim(claim) -> FactCheck`** — the agentic loop, up to `max_steps`
   iterations:
   - `search(query)` via `SearchRouter` (first iteration query = the claim
     text; later queries come from the model).
   - The model sees the claim plus evidence gathered so far and returns **either**
     `{action: "search", query: "<refined query>"}` to gather more, **or** a
     final `{verdict, confidence, rationale, evidence_used}`.
   - The loop stops at the first final verdict, or when `max_steps` is reached —
     in which case the verdict is forced to `unverified`.
   - `search_steps` records how many search iterations ran (a showcase signal
     for the "agentic loop" story and for cost reasoning).

3. **`check(claims: list[str]) -> list[FactCheck]`** — verify raw claim strings
   (used by the `claims` MCP input mode, which skips extraction); each string is
   wrapped into a `Claim` (`timestamp_s=None`) and run through `verify_claim`.

`verify_claim` takes a `Claim`; `extract_claims` produces `Claim`s with located
timestamps, `check` wraps bare strings — the two entry paths converge on one
verification method.

**Verdict vocabulary:** `supported | refuted | misleading | unverified`, where
`misleading` = the claim is technically true but omits context that changes its
meaning, and `unverified` = insufficient evidence within the step budget.

## Schema Additions (`schemas.py`)

```python
class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    MISLEADING = "misleading"
    UNVERIFIED = "unverified"

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
    confidence: float | None = None      # model-reported, 0..1
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)
    search_steps: int = 0

# AnalysisReport gains:
    fact_checks: list[FactCheck] = Field(default_factory=list)

# JobOptions gains:
    fact_check: bool = False
```

`SearchResult` (internal to `search/`, not part of the report) is the raw shape
a provider returns — `{title, url, snippet, content?}` — normalized into
`Evidence` when cited.

## Search Subpackage

```python
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None

class SearchProvider(ABC):
    name: str
    async def is_available(self) -> bool: ...
    async def search(self, query: str, k: int) -> list[SearchResult]: ...
```

- **`TavilyProvider`** — POSTs to the Tavily API via `httpx` (no vendor SDK
  needed, mirroring the lean provider clients). `is_available()` is
  `True` iff the configured API-key env var is set.
- **`DuckDuckGoProvider`** — keyless, via `ddgs`; `is_available()` is always
  `True` (best-effort). The out-of-the-box default so the feature runs with no
  configuration.
- **`FakeSearch`** — canned/scripted results (including empty-result and error
  cases) for the network-free suite.
- **`SearchRouter`** — reads the ordered `search:` candidate list from config,
  returns the first provider whose `is_available()` is true, and raises
  `SearchError` if none are available (which, inline, degrades the stage).

## MCP Tool: `fact_check_claims` (sixth tool)

Registered in `mcp_server/server.py`; execution in a new
`mcp_server/runtime.py` method driving the shared `FactChecker`.

- **Args:** `job_id: str | None = None`, `url: str | None = None`,
  `claims: list[str] | None = None`, plus `quality` and `language`
  (defaults matching `analyze_video`). Search caps come from config.
- **Validation:** exactly one of `job_id` / `url` / `claims` must be provided;
  otherwise a structured `{status: "failed", reason}` result (phase-2 error
  style), never a raw exception.
- **`job_id`** → load the report from `JobStore`, run `extract_claims` +
  verification over it, return `list[FactCheck]`, and persist the results back
  onto the stored report so a later `get_report` includes them.
- **`url`** → run the full pipeline with `fact_check=True` and return the
  complete `AnalysisReport` (including `fact_checks`).
- **`claims`** → run `FactChecker.check(claims)` directly and return
  `list[FactCheck]`; no video, no `JobStore` row.

## Config

`config/models.yaml` additions:

```yaml
tasks:
  factcheck:
    cheap:    ["ollama/llama3.1:8b", "openai/gpt-4o-mini"]
    balanced: ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]
    best:     ["anthropic/claude-sonnet-5"]

fact_check:            # loop caps (cost control)
  max_claims: 8
  max_steps: 3
  results_per_search: 5

search:                # ordered candidates; first available wins
  candidates: ["tavily", "duckduckgo"]
  tavily:
    api_key_env: TAVILY_API_KEY   # env var NAME, never a literal key
```

Model IDs and API keys continue to live only in config/env, never in code,
consistent with phase 1.

## FastAPI Adapter

Add the `fact_check: bool = False` field to the `POST /jobs` request model and
pass it into `JobOptions`. No new route, no SSE change (fact-check progress, if
surfaced, rides the existing `StageEvent` channel as ordinary stage events).
Everything else in `src/api/` is unchanged this phase.

## Error Handling

- **No search provider available** (`SearchError` from the router): inline, the
  agent raises and the pipeline appends `fact_check` to `degraded_stages`; the
  report ships without fact-checks. In the MCP tool, mapped to
  `{status: "failed", reason}`.
- **A single claim's verification errors** (search timeout, unparseable model
  output after retry): that claim gets a `FactCheck` with verdict `unverified`
  and a rationale noting the failure; other claims proceed. One bad claim never
  sinks the batch.
- **Step budget exhausted:** verdict forced to `unverified` (not an error).
- **Model-call failures** inside the loop use the existing router retry +
  candidate fallback; the trace records `fallback_from` as usual.

## Testing

Network-free default suite via `FakeSearch` + the existing `FakeProvider`; the
`FactChecker` takes its model router and search router by injection, and the
pipeline is built through the same `pipeline_factory` seam the other adapters
use.

- **FactChecker unit tests:**
  - `extract_claims` returns atomic claims with located timestamps;
  - `verify_claim` reaches a final verdict within budget;
  - budget exhaustion forces `unverified`;
  - each verdict path (`supported` / `refuted` / `misleading` / `unverified`)
    from scripted evidence + model responses;
  - a refine step (`{action:"search"}`) triggers a second search;
  - a single claim's search error yields an `unverified` `FactCheck` without
    aborting the batch.
- **SearchRouter tests:** Tavily key set → Tavily chosen; unset → DuckDuckGo;
  none available → `SearchError`.
- **Agent tests:** no-op (and not degraded) when `fact_check` is off; populates
  `report.fact_checks` when on; degrades the stage on `SearchError`.
- **MCP tool tests:** each input mode (`job_id` / `url` / `claims`) through the
  in-process `FastMCP` instance; mutual-exclusivity violation →
  `{status:"failed", reason}`; `job_id` mode persists results back to the store.
- **`@pytest.mark.slow`:** one real keyless DuckDuckGo search, asserting the
  provider returns well-formed results (network-gated, out of the default run).

## Dependencies

Add `ddgs` (keyless DuckDuckGo search) to `requirements.txt`. Tavily is called
through the already-present `httpx`, so it needs no new dependency. No other
additions.

## Roadmap (remaining phases, unchanged)

- **Phase 3 — Visual Agent:** slide/code/chart detection + OCR between
  Transcriber and Synthesizer (deferred past phase 4 by choice).
- **Phase 5 — Live streams:** chunked rolling summaries on the existing event
  channel.
- **Phase 2e — Deployment re-fit:** adapt Docker/compose/nginx/AWS tooling from
  `ollama-integration` to FastAPI + SPA (+ the MCP HTTP transport).
- **Evaluation re-wire:** point `src/evaluation/` (BERTScore/quality metrics) at
  the new pipeline's reports.
- **Fact-check UI:** surface `fact_checks` in the SPA report view (deferred from
  this phase).

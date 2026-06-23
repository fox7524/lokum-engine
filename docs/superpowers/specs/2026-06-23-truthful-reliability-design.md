# Truthful Reliability Design

## Goal

Make `lokum-engine` stop hiding failure.

This milestone is about one rule:

- the engine must not lie

In practice:

- no silent failures
- no fake empty results when retrieval actually broke
- no training on obviously invalid datasets
- no swallowed persistence failures that leave state half-saved

This is the first step toward an industry-standard engine because trust comes before quality tuning, speed work, or production scaling.

## Why This First

Current reliability risks cluster around three themes:

- RAG can return empty outputs when the real issue is query or index failure
- fine-tune can accept malformed dataset inputs too quietly
- persistence and validation paths swallow too many exceptions

If the engine cannot report failure truthfully, users cannot trust retrieval quality, fine-tune quality, or benchmark results.

## Scope

This milestone includes:

- RAG query error truthfulness
- RAG ingest and persistence error truthfulness
- fine-tune dataset preflight validation
- strict malformed JSONL handling
- core logging cleanup on reliability-critical paths
- regression tests for failure reporting and validation behavior

This milestone does not include:

- retrieval quality upgrades
- reranking
- performance optimization
- production service architecture
- benchmark framework
- advanced training orchestration

## Design Principles

- Fail clearly, not silently.
- Distinguish "no result" from "broken operation".
- Fail fast on invalid training inputs.
- Treat persistence failures as real failures.
- Keep changes focused and backward-aware.

## RAG Design

### Query Truthfulness

`RAGEngine.query()` and `RAGEngine.query_with_sources()` should stop collapsing most failures into empty results.

Target behavior:

- real retrieval miss -> empty result, no error
- operational failure -> empty result plus explicit error state

Minimum standard:

- always set `last_error` on non-abort failures
- stop relying on `print()` as the only signal
- keep the current user-facing behavior stable where possible, but expose failure clearly

### Ingest Truthfulness

Ingest should report when extraction, embedding, or persistence fails in a structured way.

Target behavior:

- valid empty content is distinguishable from broken processing
- caller can inspect failures after ingest
- save/state failures do not look like success

### Persistence Truthfulness

Critical write paths should not swallow exceptions.

Priority paths:

- index save
- metadata/state save
- delete-state updates
- destructive reset paths

Rule:

- if persistence is required for correctness and durability, failure should surface as failure

## Fine-Tune Design

### Dataset Preflight

Before training starts, validate:

- dataset directory exists
- required files exist
- files are non-empty
- rows match expected structure
- `text` field exists where required

This keeps bad launches from failing deep inside `mlx_lm`.

### Strict JSONL Handling

Malformed JSONL should not silently turn into plain text records by default in reliability mode.

Target behavior:

- strict mode raises or returns a clear validation failure
- non-strict compatibility mode can remain available if needed later

### Validation Truthfulness

Validation setup should stop swallowing presplit failures.

Rule:

- if validation preparation fails, the engine should say so explicitly

## Logging Design

Replace reliability-critical `print()` calls with structured logging.

First-pass goal:

- use Python logging in core reliability paths
- make warning/error events visible to library users and testable

This is not a full observability system yet.

## Testing

Add focused regression coverage for:

- query failure surfaces explicit error state
- ingest/persistence failures surface clearly
- malformed fine-tune dataset is rejected
- missing or empty train/valid dataset fails preflight
- validation setup failure does not disappear silently

Tests should stay narrow and behavioral.

## Acceptance Criteria

This milestone is complete when:

- RAG failure paths no longer masquerade as simple empty results
- critical persistence failures are surfaced
- fine-tune launch rejects invalid datasets before subprocess execution
- malformed JSONL no longer silently mutates data in strict reliability flow
- reliability-critical paths use logging instead of raw `print()`
- regression tests cover the new guarantees

## Risks

- some callers may currently depend on silent fallback behavior
- exposing failures may reveal hidden issues that were previously masked
- changing ingest/query return semantics too aggressively could cause avoidable churn

Mitigation:

- preserve current shapes where practical
- add explicit error fields/state before making larger interface changes

## Recommended Next Step

After this milestone:

- move to RAG correctness hardening
- then retrieval quality/eval
- then developer UX

This keeps the engine trustworthy before making it richer.

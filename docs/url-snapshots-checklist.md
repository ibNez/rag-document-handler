# URL Snapshots Implementation Checklist (Option A: PDF + MHTML)

Status legend: [ ] Not started · [~] In progress · [x] Done
Constraint: No in-app migrations. Any DDL is executed manually outside the app.

Last updated: 2025-08-19

## Quick summary
Capture two artifacts per crawl (PDF for human reference, MHTML for fidelity), store under a stable snapshots dir, register them in `documents`, link via `url_snapshots`, and set Milvus `document_id` to the snapshot id for point-in-time traceability.

---

## Phase A — Design and Documentation
- [x] A1. Document snapshot approach
  - Update docs to describe PDF+MHTML snapshots, storage layout, and Milvus mapping.
  - Acceptance: Docs include “URL Snapshots” section, storage paths, schema, and data flow.
- [x] A2. Define storage layout
  - Decide base dir: `uploaded/snapshots/{url_id}/{UTC_ISO8601}/page.pdf`, `page.mhtml`, plus `metadata.json`. Base path is configurable via `SNAPSHOT_DIR`.
  - Acceptance: Path convention documented and agreed. Default base dir is `uploaded/snapshots`.

## Phase B — Schema (manual, outside the app)
- [ ] B0. ALTER `urls` table (per-URL controls)
  - Add columns: `snapshot_enabled BOOLEAN DEFAULT FALSE`, `snapshot_retention_days INTEGER`, `snapshot_max_snapshots INTEGER`.
  - Acceptance: Columns exist; verified via psql. No app code creates/drops them.
- [ ] B1. Create `url_snapshots` table
  - Columns: `id` (UUID PK), `url_id` (UUID FK→urls.id), `snapshot_ts` (TIMESTAMP), `pdf_document_id` (VARCHAR), `mhtml_document_id` (VARCHAR), `sha256` (TEXT), `notes` (TEXT).
  - Acceptance: Table exists; verified via psql. No app code creates/drops it.
- [ ] B2. Confirm `documents` table usage
  - Reuse existing `documents` for stored files (PDF and MHTML). No schema change.
  - Acceptance: Docs show linkage via `url_snapshots`.

## Phase C — Configuration
- [ ] C1. Add .env settings (read in Config)
  - `SNAPSHOT_DEFAULT_ENABLED=true|false` (default for new URLs; actual toggle is per-URL)
  - `SNAPSHOT_DIR=./uploaded/snapshots`
  - `SNAPSHOT_FORMATS=pdf,mhtml`
  - `SNAPSHOT_TIMEOUT_SECONDS=60`
  - Acceptance: .env template updated; Config reads values.

## Phase D — Dependencies & Environment
- [ ] D1. Add Python deps
  - `playwright`, `pytest-playwright` (for tests).
  - Acceptance: dependencies added to project manifest.
- [ ] D2. Install browsers (local manual step)
  - Run: playwright install chromium
  - Acceptance: Documented in README; can render pages headlessly.

## Phase E — Snapshot Capture Service (isolated module)
- [ ] E1. Implement capture function
  - Use Playwright (Chromium) to load URL (wait_until=networkidle), export PDF, export MHTML via CDP `Page.captureSnapshot`.
  - Return paths and SHA-256 of canonical text.
  - Acceptance: Unit tests for happy path and timeouts.
- [ ] E2. File system handling
  - Create directories, write files atomically, compute hashes, write `metadata.json`.
  - Acceptance: Tests verify paths, files, and `metadata.json` contents.

## Phase F — Persistence & Linking
- [ ] F1. Register files in `documents`
  - Create two entries (PDF and MHTML) with `content_type`, `file_size`, metadata (path, hash, url_id, snapshot_id).
  - Acceptance: Helper returns both `document_id`s.
- [ ] F2. Insert `url_snapshots` row
  - Insert with `url_id`, `snapshot_ts` (UTC now), `pdf_document_id`, `mhtml_document_id`, `sha256`.
  - Acceptance: Row is created and queryable.

## Phase G — Ingestion Pipeline Integration
- [ ] G1. Hook snapshots into URL crawl
  - On successful crawl and text extraction, call snapshot service before embedding.
  - Acceptance: Artifacts exist and are linked for each successful crawl.
- [ ] G2. Update Milvus mapping
  - Set `document_id` to `snapshot_id` for URL embeddings; keep `source=url`, `category="url"`.
  - Acceptance: New Milvus inserts use `snapshot_id` consistently.
- [ ] G3. Backward compatibility flag
  - If URL `snapshot_enabled=false`, revert to previous behavior (`document_id=url_id`). Use `SNAPSHOT_DEFAULT_ENABLED` only as a default for new URLs.
  - Acceptance: Toggle works without errors.

## Phase H — UI and UX
- [ ] H1. Latest snapshot links in URL list/modal
  - Add “View PDF” and “Download MHTML” for latest snapshot.
  - Acceptance: Links visible when available.
- [ ] H2. Serve files locally (read-only)
  - Routes or static serving for `SNAPSHOT_DIR` files.
  - Acceptance: Clicking links opens/downloads artifacts.

## Phase I — Retention & Ops
- [ ] I1. Retention policy
  - Per-URL: `snapshot_retention_days` and/or `snapshot_max_snapshots`; scheduled cleanup job.
  - Acceptance: Cleanup deletes oldest snapshots safely (files + rows).
- [ ] I2. Error handling & retries
  - Timeouts, non-200 loads, heavy pages; log errors; safe fallbacks.
  - Acceptance: Crawl continues; failures are visible in logs/status.

## Phase J — Testing
- [ ] J1. Unit tests
  - Snapshot service (PDF, MHTML, metadata), persistence helpers, Milvus id mapping logic.
  - Acceptance: Tests pass locally.
- [ ] J2. Integration tests
  - End-to-end: crawl → snapshot → documents rows → url_snapshots row → Milvus insert with `snapshot_id`.
  - Acceptance: Green tests with local services.
- [ ] J3. Manual QA checklist
  - Steps to verify UI links, file existence, and Milvus records.
  - Acceptance: Checklist stored in `docs/QA.md`.

## Phase K — Documentation & Release
- [ ] K1. Update docs
  - Architecture and database-schema docs updated for snapshots, linking, Milvus mapping, and ops.
  - Acceptance: Reviewed and committed.
- [ ] K2. Rollout plan
  - Manual steps noted: create `url_snapshots` table via psql, install Playwright browsers, update `.env`.
  - Acceptance: Clear runbook in README.

---

## Decision log
- Store two artifacts per crawl: PDF (human-friendly) + MHTML (high-fidelity, single-file).
- Files on disk; metadata and linkage in PostgreSQL; embeddings in Milvus.
- No in-app migrations; DDL executed manually outside the application.

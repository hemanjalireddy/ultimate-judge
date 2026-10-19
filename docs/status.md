# Implementation status

## Rate-limit recovery

Per-model pacing, rate-limit countdowns, persisted retry times, and same-comparison
resumption are implemented. Old API-error gaps are retried and their history retained;
completed judgments are reused. Verbosity is included in the home-page defaults.
Validation: **46 passed, 1 skipped** with mocked rate-limit and resume checks.

## Home-page workflow

The home page now uploads questions only, loads local movie facts automatically,
provides **Run test**, shows generated answers side by side, then provides
**Run judge** and inline results. Existing manual UI is preserved in `legacy_app.py`.
Keyword fact lookup is deterministic and inspectable, not a full constraint solver.
Validation: **42 passed, 1 skipped**, including a UI test that clicks both buttons
with mocked provider output and verifies stale results clear when inputs change.

## Docker fix verified

Docker and CI now pin Poetry 2.4.1, matching the tool that generated the lock file.
Poetry 2.2.1 rejected its content hash. The rebuilt image, `docker compose up -d`,
and the container's Streamlit health check now pass. Earlier Docker-unavailable
notes below describe previous attempts, not the current state.

## Question-upload workflow update

Validation: `poetry run pytest -q` — **40 passed, 1 skipped** (opt-in live Groq
test); `docker compose config --quiet` passed. Live API and Docker-engine
limitations below still apply.

- Streamlit now accepts TXT/JSON question files, resolving movie IDs from an
  uploaded or existing TMDB CSV. Plain questions use user-selected shared options.
- `questions.py` defines scenario-only inputs; `dataset.py` now outputs
  `data/scenarios.json` with no template answers. Existing files are preserved.
- Answer-model selection is independent of judge selection in both UI and CLI.
- Partial generation is saved as `.progress.json`. Only a complete set of model
  answers becomes a frozen benchmark; live judging rejects template/partial inputs.
- Examples: `examples/questions.txt` and `examples/questions.json`.

## Verification on 2026-10-14

- `poetry run pytest -q`: **32 passed, 1 skipped**. The skipped check requires a
  Groq key and explicit live-test opt-in.
- Full real-input/offline execution: **30 cases, 1,800 mock observations**.
  These validate the runner, not model behavior.
- Bundled synthetic demo: **120 mock observations**; software metrics regression
  fixture is checked by tests.
- `docker compose config --quiet`: passed.
- `poetry check`: passed with legacy metadata deprecation warnings.
- Local Streamlit process started; HTTP health endpoint returned `ok`.
- `docker compose build`: failed to connect to the unavailable Docker engine.

New commands are documented in the README. Changed legacy files: `app.py`,
`models.py`, `runner.py`, Poetry configuration/lock, Docker configuration, ignore
files and README. Added modules: `provider.py`, `benchmark_models.py`, `dataset.py`,
`benchmark.py`, `metrics.py`, and `pages/1_Benchmark.py`. Added supporting files:
`configs/`, `examples/`, `tests/`, `docs/`, `benchmark-manifest.json`, and the
GitHub Actions workflow. No existing experiment data was replaced.

## Implemented

- Existing interactive workflow and version-1 JSON model retained.
- Environment-only key access; provider retry/backoff and safe error messages.
- Versioned benchmark schema, frozen inputs, snapshot hashes, constraint checks.
- Thirty real-data scenario instances constructed from six authored templates.
- Separate candidate-generation command, checkpointed between generations.
- Four experimental conditions: baseline position/self-preference, direct
  original-versus-expanded and original-versus-formatted comparisons.
- Three trials per order by default; seeded order randomization; canonical IDs.
- Core metrics, case-cluster bootstrap, usage metadata and generator/judge table.
- Sequential budgeted CLI/UI runner, file lock, atomic saves and resumption.
- Offline fixture, dashboard filters, raw judgment explorer, Plotly chart.
- Offline tests, opt-in live test, GitHub Actions test/build/health workflow.

## Not yet verified or completed

- No Groq credentials were available during implementation. Model-listing and
  real generation/judging are unverified; no empirical model findings exist.
- Docker client is installed, but the engine is unavailable. Compose validates;
  image build and container health have not been verified locally.
- CI workflow is written, not yet executed on GitHub.
- Thirty case instances are deterministic and inspectable, not human-annotated.
  Equivalence reviews and human quality labels remain pending.
- No configurable parallelism, cost estimator, full run-comparison/regression UI,
  or advanced stylistic transformations. Current concurrency is exactly one.
- Human labels can be supplied through the versioned JSON schema; the UI currently
  provides transformation review, not a full annotation platform.

## Roadmap

1. Verify both hosted models using an opt-in live smoke test.
2. Generate and review frozen answers for the real 30-case snapshot.
3. Run counterbalanced experiments; inspect failures before publishing findings.
4. Add matched run comparison with uncertainty for differences, then configurable
   concurrency/cost estimates if real usage justifies them.

An introductory post can describe the question and engineering work in progress.
Do not publish the mock rates as findings or claim all phases are complete.

# Fine Print: AI M&A agreement review and deal simulation

Runs on **local open models through Ollama** (tuned for a 16 GB NVIDIA T4) or on **Google Gemini**.
Switch with one line in `utils.py`.

Reads M&A agreements (merger agreements, LOIs, SEC filings as `.txt`), flags risky fine print,
extracts the deal economics, simulates post-merger cash flow under stress, scores the deal on a
transparent 100-point rubric, and recommends countermeasures, in a Streamlit web app.

**Gemini reads the agreement and exercises judgement. Deterministic Python does the money.**
Every number in the simulation and the scorecard traces back to a formula, not a generation.

---

## Quick start

```bash
cd ma_fineprint_engine
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Open **`utils.py`**. For local models leave `LLM_PROVIDER = "ollama"` (see the T4 section below). For Gemini set
   `LLM_PROVIDER = "gemini"` and paste your key between the quotes: `GEMINI_API_KEY = "AIza..."`.
   That is the only file you edit. Users of the app are never asked for a key or a model.
2. `streamlit run app.py`. The top-right corner says **Running locally on Ollama (model)** or
   **Connected to Gemini (model)** when everything works.
   If something is wrong (missing quotes, rejected key, unknown model, no internet) the start screen says
   exactly what, and the app picks up a fixed `utils.py` when you reload the page.

Optional checks:

```bash
pytest -q                                   # 90 tests, offline, ~20 s
python scripts/smoke_test.py --live         # one real Gemini run on the sample agreement, in the terminal
```

Without a key the app runs as an **offline preview**: chunking, retrieval, pattern checks, simulation,
scoring and charts are real; the summary, AI review and recommendations show placeholders.

### Your dataset

Three ways in, all on the start screen:

- **Library**: put `.txt` files in `data/raw/` (sub-folders are fine); they appear automatically.
- **Files or .zip**: upload one or many `.txt` files, or a `.zip`. MAUD's `data.zip` works as is:
  its 152 agreements are listed (named by their parties, searchable) and the CSV files are skipped.
- **Folder**: upload a whole folder of `.txt` files.
Documents that were split into numbered, overlapping pieces (`contract_0.txt_1.txt ... contract_0.txt_410.txt`)
are **stitched back into one document** (overlaps removed) before analysis, so a contract split across
410 files is reviewed as one agreement. Uploading several pieces in the app does the same.

`scripts/ingest_dataset.py` can pre-build a search index for scripts; the app does not need it.

---

## Run locally on an NVIDIA T4 with Ollama

`utils.py` ships with `LLM_PROVIDER = "ollama"`. On any Linux machine with a T4 (cloud VM, Colab, Kaggle):

```bash
bash scripts/setup_t4.sh             # installs/configures Ollama, pulls the models, installs Python packages
python3 scripts/benchmark_ollama.py  # speed check: both models must show "100% on GPU"
bash scripts/start_app.sh            # app on port 8501; add --public for a shareable https link
```

On **Google Colab** open `notebooks/run_on_colab_t4.ipynb`, choose a T4 runtime and run the cells: it uploads the
project, runs the setup, benchmarks, and prints a public link to the app.

### What the T4 setup does and why

| Setting | Value | Why |
|---|---|---|
| Chat model | `qwen3:14b` (Q4_K_M, ~9.3 GB) | Strongest instruction-following model that fits entirely in 16 GB; `qwen3:8b` is the faster preset |
| Embeddings | `nomic-embed-text` (~0.3 GB) | Semantic search on the same GPU; falls back to local TF-IDF if missing |
| Context | 16,384 tokens | Model + 8-bit KV cache for two parallel requests fit with headroom |
| `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` | on | Halves context memory with negligible quality loss |
| `OLLAMA_NUM_PARALLEL=2` | 2 | Two requests decode together, close to double throughput on a memory-bound GPU |
| `OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=30m` | | Chat and embedding models stay loaded between steps |
| Thinking | off (`think: false`) | Much faster; prompts are explicit enough for extraction and review |

### How the engine adapts to a 16k window

Gemini can read a whole 50,000-word agreement at once; a local model cannot. With Ollama the engine budgets every
prompt (about 6,300 words of excerpts per request) and leans on retrieval:

- **Executive summary**: the cover page plus the best passages for nine summary topics, instead of the full text.
- **Deal-term extraction**: the most relevant passages for 16 deal-term queries, round-robin, within the budget.
- **Risk review**: passages behind each pattern-check hit first, then retrieved passages, within the budget.
- **Recommendations**: a compact version of the flags and figures.
- Each prompt is checked against the window before sending; whole excerpts are dropped from the end if needed,
  so Ollama never silently truncates the start of a prompt.

Structured output uses Ollama's `format` (JSON-schema constrained decoding in llama.cpp); the schema is also written
into the prompt so the model can read the field descriptions. Every schema has been checked against llama.cpp's
JSON-schema-to-grammar converter.

A full review makes about 13 model calls. Expect roughly 8-15 minutes for a long agreement with `qwen3:14b` on a T4
(about half that with `qwen3:8b`); `scripts/benchmark_ollama.py --full` measures it on your machine.

**Troubleshooting:** out-of-memory errors: set `OLLAMA_CONTEXT = 12288` or `OLLAMA_MODEL = "qwen3:8b"`. "Model not
installed": `ollama pull <name>`. Anything less than 100% on GPU in the benchmark means CPU offload; use the same fixes.

## Where the AI model is used and where it isn't

| # | Module | Gemini? | How |
|---|--------|---------|-----|
| 1 | Ingestion & RAG | No | Stitch split files, clean, clause-aware 500-word overlapping chunks, local embeddings + BM25 fused with Reciprocal Rank Fusion |
| 2 | Executive summary | **Yes** | Length adapts to the document. Very large documents use map-reduce |
| 3 | Cash-flow simulation | No | Monthly simulator, 5 stress scenarios, 2,000-run Monte Carlo |
| 3a | Deal-term extraction | **Yes (RAG)** | Multi-query retrieval, then Gemini fills a typed schema with evidence quotes. Feeds the simulator |
| 4 | Confidence scorecard | No | 16 explicit criteria in 4 weighted categories, each with the reason for every deduction |
| 5 | Charts | No | Plotly: scenario cash paths, Monte Carlo range, value vs drains, margins, closing waterfall |
| 6 | Risk flagging | **Rules + Gemini (RAG)** | 25 pattern checks (run over the whole document, so clauses spanning passages are caught) + numeric anomaly detection, then Gemini reviews each of 10 risk categories over retrieved clauses |
| 7 | Recommendations | **Yes** | Renegotiation, escrow, pricing or walk-away, grounded in computed sizing figures |
| + | Ask the agreement | **Yes (RAG)** | Q&A answered only from retrieved passages, with citations |

A full analysis makes about 13 Gemini calls (risk categories run in parallel). Identical requests are
answered from a disk cache (`.cache/llm`). Changing assumptions in the Cash flow tab re-runs the
simulation and scorecard without calling Gemini.

### How Gemini returns structured data

Every output is a Pydantic model (`ma_engine/models/llm_outputs.py`) converted to a JSON schema and sent as
`response_json_schema`, so Gemini returns JSON in that shape. Results are validated; invalid JSON is sent back
for correction (up to 3 attempts). If Gemini ever rejects a schema as too complex, the client falls back to
JSON mode with the schema in the instructions. Rate limits and server errors are retried with backoff.
Document text is treated as untrusted data.

### Settings

`utils.py`: `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.8-flash`), `GEMINI_THINKING_LEVEL`
(`""`, `low`, `medium`, `high`). Advanced options (parallel calls, chunk size, embeddings, offline mode)
can be set as environment variables or in a `.env` file; see `.env.example`. On a free-tier key, set
`MA_MAX_PARALLEL=2` if you hit rate limits.

---

## Project layout (one purpose per file)

```
app.py                              Streamlit entry point (layout and wiring only)
utils.py                            The Gemini API key and model - the only file you edit
ma_engine/
  config.py                         All settings; env-var overrides
  pipeline.py                       Orchestrates the seven modules
  models/
    documents.py                    Document, Chunk, RetrievedChunk
    llm_outputs.py                  Schemas Gemini must fill (summary, terms, flags, advice, Q&A)
    finance.py                      Simulator inputs/outputs
    scoring.py                      Scorecard models
    report.py                       The full AnalysisReport
  ingestion/
    loader.py                       Read .txt files / uploads, encoding fallback, stable doc ids
    reassembler.py                  Stitch numbered, overlapping pieces back into one document
    archives.py                     Unpack uploaded datasets (.zip, folders), skip non-text files
    metadata.py                     Party names from the cover page, for readable labels
    cleaner.py                      Strip EDGAR/HTML, page numbers, running headers, re-flow lines
    sections.py                     Detect ARTICLE / Section headings for chunk breadcrumbs
    chunker.py                      500-word overlapping, clause-aware chunking
    preprocess.py                   raw -> cleaned Document + chunks
  retrieval/
    tokenizer.py                    Tokenizer for lexical search
    bm25.py                         Okapi BM25
    embedder.py                     sentence-transformers or TF-IDF/LSA embeddings
    index_store.py                  Vector + BM25 index, save/load
    indexer.py                      Build an index from documents
    retriever.py                    Hybrid search (RRF), multi-query, neighbours, context formatting
  llm/
    base.py                         Client interface + usage tracking
    gemini_client.py                Gemini client (JSON schema output, repair loop, retries, caching)
    ollama_client.py                Local models via Ollama (schema-constrained JSON, context budgeting)
    connection.py                   Checks Gemini key / Ollama server and models before any analysis
    mock_client.py                  Offline stand-in for tests and demo mode
    factory.py                      Real vs mock selection
    prompts.py                      Every prompt in one place
    schema_utils.py                 Pydantic -> Gemini-safe JSON schema
    response_cache.py               Disk cache of responses
  analysis/
    risk_taxonomy.py                10 risk categories: retrieval queries + market-practice benchmarks
    rule_checks.py                  25 deterministic clause checks
    anomaly_detector.py             Inconsistent KPI figures, ARR vs GAAP revenue
    risk_flagger.py                 Rules + anomalies + Gemini review, merged and ranked
    summarizer.py                   Adaptive executive summary (single pass or map-reduce)
    extractor.py                    Deal-term extraction with multi-query RAG
    scoring.py                      100-point rubric matrix
    advisor.py                      Mitigation strategy
    qa.py                           Ask-the-agreement
  finance/
    input_builder.py                DealTerms + user edits -> simulator inputs (tracks sources)
    simulator.py                    Monthly cash-flow engine
    metrics.py                      Liquidity / NPV / payback metrics
    scenarios.py                    Standard stress scenarios
    monte_carlo.py                  Probabilistic cash range and breach probability
    mitigation_math.py              Escrow sizing, price headroom, synergy break-even
  visualization/
    theme.py                        Colour-blind-safe palette and layout
    charts.py                       All Plotly charts
  reporting/markdown_report.py      Downloadable Markdown memo
  ui/
    theme.py                        Visual identity: palette, type, Streamlit styling
    components.py                   Reusable page elements (masthead, score seal, figures)
    landing.py                      Start screen: choose or upload an agreement
    library.py                      Bundled samples + data/raw documents
    assumptions.py                  Editable financial assumptions form
    state.py                        Session state
    views/document.py               1 Document & RAG: processing steps, live retrieval search, extracted terms
    views/overview.py               2 Executive summary
    views/financials.py             3 Cash flow simulation and editable assumptions
    views/scorecard.py              4 Confidence score and rubric matrix
    views/growth.py                 5 Growth & profitability trajectory
    views/risks.py                  6 Fine print risks and number anomalies
    views/advice.py                 7 Mitigation strategy
    views/ask.py                    Ask the agreement (Q&A with citations)
  utils/formatting.py               Money / percent formatting
scripts/
  setup_t4.sh                       Install and tune Ollama on a T4, pull models, install packages
  start_app.sh                      Start the app (--public for a shareable link)
  benchmark_ollama.py               Speed and GPU-placement check for the local model
  ingest_dataset.py                 Index a folder of .txt files
  smoke_test.py                     End-to-end run (mock or --live)
notebooks/run_on_colab_t4.ipynb     One-click run on a free Colab T4
sample_data/                        Cisco/Acacia agreement (from your dataset) + a fictional one with planted risky clauses
tests/                              90 tests (ingestion, retrieval, finance, scoring, Gemini client, pipeline, app)
```

---

## The simulation in brief

Month 0 is closing. Closing cash = acquirer cash + new debt + target cash - cash consideration -
transaction costs - change-of-control payments - accelerated target debt (with premium). Each month:
revenue and costs grow on each company's path; synergies ramp linearly (after any delay) to run-rate;
integration costs are spread over the integration period; interest on acquisition and remaining target
debt; tax on positive pre-tax profit; capex as % of revenue; one-offs (hidden liabilities net of escrow
recovery, earnout). Scenarios: **Base, Delayed Integration, Hidden Liabilities, Debt Acceleration + Rate
Shock, Severe Downside**. Monte Carlo varies synergy realisation, delay, cost overrun, revenue, rates,
liabilities and the earnout (seeded, reproducible).

Merger agreements rarely contain the acquirer's own financials or its cash covenant, so those start as
illustrative estimates (marked * in the Assumptions form on the Cash flow tab). Enter real figures
there and press **Update simulation**.

## Scorecard rubric (default weights)

| Category (points) | Criteria |
|---|---|
| Financial Risk (30) | Base-case liquidity headroom 10 · Severe-downside survival 8 · Monte Carlo breach probability 6 · Leverage & debt acceleration 6 |
| Legal Liability (25) | Termination fee structure 7 · MAE balance 6 · Indemnity & escrow 6 · Open legal red flags 6 |
| Regulatory Compliance (20) | Regulatory burden & remedies 7 · Outside-date runway 5 · Financing certainty 4 · Regulatory/financing flags 4 |
| Synergy Realization (25) | Integration payback 7 · Synergy NPV vs price 6 · Delay sensitivity 6 · Synergy break-even for liquidity 6 |

Ratings: **A** >= 80 · **B** 65-79 · **C** 50-64 · **D** < 50. Weights are in `config.py` (`ScoringWeights`).

---

## Limitations

- Input is plain text. Convert PDFs/Word first (keep the text layer; tables may need checking).
- The analysis takes the **acquirer's** perspective throughout.
- The simulator is a planning model (no working capital, D&A or detailed tax). It is meant to compare
  scenarios and size protections, not to replace a full merger model.
- Decision support only, not legal or investment advice. Always verify flags against the source clauses
  (each flag links to its excerpt).

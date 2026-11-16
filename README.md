# Fine Print: AI M&A agreement review and deal simulation

Fine Print reads M&A agreements (merger agreements, LOIs, SEC filings as `.txt`), flags risky fine print, extracts the deal economics, simulates post-merger cash flow under stress, scores the deal on a transparent 100-point rubric and recommends countermeasures. It runs as a Streamlit web app.

- **The AI reads and judges. Plain Python does the money.** Every number in the simulation and the scorecard comes from a formula, not from a model.
- **Two ways to run the AI**, switched with one line in `utils.py`:
  - `ollama`: open models (`qwen3:14b` + `nomic-embed-text`) on your own GPU. No API key, and nothing leaves the machine. Tuned for a 16 GB NVIDIA T4.
  - `gemini`: Google's Gemini API. Needs a free API key and no GPU.

> Decision support only, not legal or investment advice. Always check flags against the source clauses.

## Contents

- [Which option should I pick?](#which-option-should-i-pick)
- [Run on Google Colab](#run-on-google-colab)
- [Run on Kaggle](#run-on-kaggle)
- [Run locally](#run-locally)
- [Using the app](#using-the-app)
- [Configuration](#configuration)
- [Tests and checks](#tests-and-checks)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Which option should I pick?

| Where | AI backend | GPU needed | Key needed | Setup time | Good for |
|---|---|---|---|---|---|
| **Colab** (free T4) | Ollama | Yes (T4, provided) | No | ~5 min | Quickest way to try the full local-model setup |
| **Kaggle** (T4 x2) | Ollama | Yes (provided) | No | ~5 min | Same, with a weekly GPU quota |
| **Local, Linux + NVIDIA GPU (16 GB)** | Ollama | Yes | No | ~10 min | Private, repeatable, offline after setup |
| **Local, Mac / Windows / no GPU** | Gemini | No | Yes | ~2 min | Lightweight use on any laptop |
| **Anywhere, no key, no GPU** | Offline preview | No | No | ~2 min | Looking around the UI and the finance engine |

Colab and Kaggle sessions are temporary. Models, files and the public link are lost when the session ends, so a fresh session repeats the setup.

> **Repo layout note.** The commands below assume the project files (`app.py`, `utils.py`, `scripts/`, ...) sit at the **root** of the repo. If they live in a sub-folder (for example `ma_fineprint_engine/`), run `cd ma_fineprint_engine` after cloning.
> Replace `YOUR_USERNAME/YOUR_REPO` with this repository's path.

---

## Run on Google Colab

### Option A: Local models on the free T4 (no key)

1. Open [colab.research.google.com](https://colab.research.google.com) and create a **New notebook**.
2. **Runtime > Change runtime type > T4 GPU > Save**.
3. Paste each block below into its own cell and run them in order.

**Cell 1: check the GPU** (should say `Tesla T4`, about 15 GB)

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2: get the code**

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

**Cell 3: install Ollama, download the models, install Python packages** (about 5 minutes the first time)

```python
!bash scripts/setup_t4.sh
```

**Cell 4: speed check.** Both models should report `100% on GPU`. If not, see [Troubleshooting](#troubleshooting).

```python
!python3 scripts/benchmark_ollama.py
```

**Cell 5: start the app and print a public link**

```python
import subprocess, re, time, pathlib

subprocess.Popen(["bash", "scripts/start_app.sh", "--public"],
                 stdout=open("start.log", "w"), stderr=subprocess.STDOUT,
                 start_new_session=True)

for _ in range(90):
    log = pathlib.Path("tunnel.log").read_text() if pathlib.Path("tunnel.log").exists() else ""
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    if m:
        print("Open the app:", m.group(0)); break
    time.sleep(2)
else:
    print("No link yet. See start.log and tunnel.log")
```

Open the printed `https://....trycloudflare.com` link. It works only while the notebook stays connected.

> The repo also ships `notebooks/run_on_colab_t4.ipynb`, which does the same steps but asks you to upload the project as a zip. Use it if you would rather not clone from GitHub. In Colab: **File > Upload notebook**.

### Option B: Gemini on Colab (no GPU needed)

1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. In Colab, open the key icon (**Secrets**), add a secret named `GEMINI_API_KEY` with your key, and turn on **Notebook access**.
3. Run:

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -q -r requirements.txt

import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")   # picked up at startup
!sed -i 's/^LLM_PROVIDER = .*/LLM_PROVIDER = "gemini"/' utils.py
```

Then run Cell 5 from Option A to start the app. `start_app.sh` prints a note that Ollama is not running; ignore it when using Gemini.

---

## Run on Kaggle

Kaggle gives you two T4 GPUs and a weekly GPU quota. The app uses one.

1. Sign in at [kaggle.com](https://www.kaggle.com) and **verify your phone number** (needed for GPU and internet access): *Settings > Phone verification*.
2. **Create > New Notebook**.
3. In the right-hand **Session options** panel:
   - **Accelerator:** *GPU T4 x2* (a single T4 also works; avoid P100, which does not support the flash-attention setting the setup uses).
   - **Internet:** **On**. Without it the model download and the public link both fail.
4. Run the cells below in order.

**Cell 1: use one GPU and check it** (this must run before anything starts Ollama)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2: get the code**

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

**Cell 3: install Ollama, download the models, install Python packages**

```python
!bash scripts/setup_t4.sh
```

**Cell 4: speed check** (both models should say `100% on GPU`)

```python
!python3 scripts/benchmark_ollama.py
```

**Cell 5: start the app and print a public link**

```python
import subprocess, re, time, pathlib

subprocess.Popen(["bash", "scripts/start_app.sh", "--public"],
                 stdout=open("start.log", "w"), stderr=subprocess.STDOUT,
                 start_new_session=True)

for _ in range(90):
    log = pathlib.Path("tunnel.log").read_text() if pathlib.Path("tunnel.log").exists() else ""
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    if m:
        print("Open the app:", m.group(0)); break
    time.sleep(2)
else:
    print("No link yet. See start.log and tunnel.log")
```

Open the printed link and keep the notebook tab open while you work.

**Kaggle notes**

- Use the interactive notebook session. A *Save & Run All* batch run ends when the cells finish, which stops the app.
- Kaggle stops idle sessions and enforces a weekly GPU quota, so stop the session when you are done.
- **Gemini instead of Ollama:** skip Cell 3 and Cell 4. Add your key under **Add-ons > Secrets** as `GEMINI_API_KEY`, then run:

  ```python
  from kaggle_secrets import UserSecretsClient
  import os
  os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
  !pip install -q -r requirements.txt
  !sed -i 's/^LLM_PROVIDER = .*/LLM_PROVIDER = "gemini"/' utils.py
  ```

  then run Cell 5.

---

## Run locally

Requires **Python 3.10 or newer** (3.11+ recommended) and git.

### 1. Get the code and install packages (all platforms)

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then pick one of the three setups below.

### 2a. Linux with an NVIDIA GPU (16 GB), local models, one command

Needs an NVIDIA driver (`nvidia-smi` must work), `curl`, and either root or `sudo`.

```bash
bash scripts/setup_t4.sh             # installs/configures Ollama, pulls both models, installs packages
python3 scripts/benchmark_ollama.py  # both models should show "100% on GPU"
bash scripts/start_app.sh            # app at http://localhost:8501
```

On a cloud VM, open port 8501 in the firewall, or run `bash scripts/start_app.sh --public` for a temporary shareable `https` link.

Alternatively, start the app in the foreground with `streamlit run app.py`.

### 2b. macOS or Windows, local models

`setup_t4.sh` is Linux-only, so set Ollama up by hand:

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Pull the models:

   ```bash
   ollama pull qwen3:14b
   ollama pull nomic-embed-text
   ```

   `qwen3:14b` needs roughly 16 GB of GPU or unified memory. On smaller machines pull `qwen3:8b` instead and set `OLLAMA_MODEL = "qwen3:8b"` in `utils.py`.
3. Keep `LLM_PROVIDER = "ollama"` in `utils.py`, make sure the Ollama app is running, and start the app:

   ```bash
   streamlit run app.py
   ```

Optional performance settings (set before starting Ollama): `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=2`. Without a discrete GPU, local models are very slow. Use Gemini (below) instead.

### 2c. Any machine, Gemini (no GPU)

1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Edit **`utils.py`** (the only file you need to edit):

   ```python
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "AIza..."        # keep the quotes
   ```

   Do not commit a real key. Alternatively leave `GEMINI_API_KEY` empty in the file and export it instead (`export GEMINI_API_KEY=AIza...`, or `$env:GEMINI_API_KEY="AIza..."` in PowerShell).
3. Start the app:

   ```bash
   streamlit run app.py
   ```

On a free-tier key, set `MA_MAX_PARALLEL=2` if you hit rate limits.

### Offline preview (no key, no GPU)

To explore the app without any AI backend, run with the offline mock. Chunking, retrieval, pattern checks, the simulation, the scorecard and the charts are real. The summary, AI review and recommendations show placeholders.

```bash
MA_DEMO=1 streamlit run app.py            # Windows PowerShell: $env:MA_DEMO="1"; streamlit run app.py
```

### Check that it works

Open <http://localhost:8501>. The top-right corner should read **Running locally on Ollama (model)** or **Connected to Gemini (model)**. If something is wrong (missing quotes, rejected key, unknown model, Ollama not running), the start screen says exactly what. Fix it and reload the page.

---

## Using the app

On the start screen, choose a document in one of these ways:

- **Bundled samples:** a Cisco/Acacia agreement and a fictional agreement with planted risky clauses (`sample_data/`).
- **Library:** put `.txt` files in `data/raw/` (sub-folders are fine). They appear automatically.
- **Upload files or a `.zip`:** one or many `.txt` files, or a zip such as MAUD's `data.zip`.
- **Upload a folder** of `.txt` files.

Agreements that were split into numbered, overlapping pieces (`contract_0.txt_1.txt`, `contract_0.txt_2.txt`, ...) are stitched back into one document automatically.

Input must be plain text. Convert PDFs and Word files first.

A full review makes about 13 model calls. With `qwen3:14b` on a T4, a long agreement takes roughly 8 to 15 minutes (about half that with `qwen3:8b`). Gemini is typically faster.

---

## Configuration

All provider and model settings live in `utils.py`:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `"ollama"` | `"ollama"` (local) or `"gemini"` (API) |
| `OLLAMA_MODEL` | `"qwen3:14b"` | Chat model. `"qwen3:8b"` is about twice as fast |
| `OLLAMA_EMBED_MODEL` | `"nomic-embed-text"` | Embedding model for document search |
| `OLLAMA_CONTEXT` | `16384` | Context window per request. Use `12288` if memory runs out |
| `OLLAMA_PARALLEL` | `2` | Parallel requests; must match `OLLAMA_NUM_PARALLEL` on the server |
| `GEMINI_API_KEY` | `""` | Your Gemini key |
| `GEMINI_MODEL` | `"gemini-3.8-flash"` | Gemini model |
| `GEMINI_THINKING_LEVEL` | `""` | `""`, `low`, `medium` or `high` |

Advanced options (parallel calls, chunk size, embeddings, offline mode, caching) are environment variables or entries in a `.env` file. Copy `.env.example` to `.env` to use them. After changing `OLLAMA_MODEL`, `OLLAMA_CONTEXT` or `OLLAMA_PARALLEL`, re-run `scripts/setup_t4.sh` (Colab/Kaggle: re-run the setup cell) so the server picks them up.

---

## Tests and checks

```bash
pytest -q                                   # offline, no key or GPU needed
python scripts/smoke_test.py                # end-to-end run on the sample with the mock model
python scripts/smoke_test.py --live         # one real run using the Gemini key from utils.py
python scripts/benchmark_ollama.py --full   # times a complete review on your GPU
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Benchmark shows less than `100% on GPU` or you get out-of-memory errors | Set `OLLAMA_CONTEXT = 12288` or `OLLAMA_MODEL = "qwen3:8b"` in `utils.py`, then re-run the setup step. On Kaggle, make sure Cell 1 (`CUDA_VISIBLE_DEVICES`) ran first |
| `nvidia-smi: command not found` / very slow answers | No GPU is attached. Colab: *Runtime > Change runtime type > T4 GPU*. Kaggle: set the accelerator, then restart the session |
| Setup or the public link fails on Kaggle | Turn **Internet** on in Session options (needs a phone-verified account) |
| "Model not installed" | `ollama pull qwen3:14b` and `ollama pull nomic-embed-text` (or the names set in `utils.py`) |
| Start screen says Ollama is not reachable | Start it: `ollama serve` (or open the Ollama app). On Colab/Kaggle re-run the setup cell. Server log: `tail -n 30 ollama.log` |
| Start screen says `utils.py` has a syntax error | The key must be inside quotes: `GEMINI_API_KEY = "AIza..."` |
| Gemini key rejected | Check it starts with `AIza` and has no spaces. Create a new one at aistudio.google.com/apikey |
| Gemini rate-limit errors on a free key | `export MA_MAX_PARALLEL=2` (or put it in `.env`) |
| No public link appears | Read `start.log` and `tunnel.log`. Re-run the start cell. Links last only while the session is alive |
| App log | `streamlit.log` in the project folder |

---

## Project layout

```
app.py                  Streamlit entry point
utils.py                Provider, model and key settings (the only file you edit)
requirements.txt        Python dependencies
.env.example            Optional advanced settings
ma_engine/              The engine
  ingestion/            Load, stitch, clean and chunk agreements
  retrieval/            Hybrid search: embeddings + BM25
  llm/                  Gemini and Ollama clients, prompts, connection checks, mock client
  analysis/             Summary, deal-term extraction, risk rules, scoring, advisor, Q&A
  finance/              Cash-flow simulator, stress scenarios, Monte Carlo, mitigation sizing
  visualization/        Plotly charts
  reporting/            Downloadable Markdown memo
  ui/                   Start screen and the app's views
scripts/                setup_t4.sh, start_app.sh, benchmark_ollama.py, ingest_dataset.py, smoke_test.py
notebooks/              run_on_colab_t4.ipynb
sample_data/            Two sample agreements
data/raw/               Put your own .txt agreements here
tests/                  Offline test suite
```

## How it works, in short

| Step | Uses the AI? | How |
|---|---|---|
| Ingestion and retrieval | No | Clean text, clause-aware 500-word overlapping chunks, embeddings + BM25 fused with Reciprocal Rank Fusion |
| Executive summary | Yes | Adapts to document length; very long documents use map-reduce |
| Deal-term extraction | Yes (retrieval-grounded) | Fills a typed schema with evidence quotes; feeds the simulator |
| Cash-flow simulation | No | Monthly model, 5 stress scenarios, 2,000-run Monte Carlo |
| Confidence scorecard | No | 16 criteria in 4 weighted categories (A >= 80, B 65-79, C 50-64, D < 50) |
| Risk flagging | Rules + AI | 25 pattern checks and numeric anomaly checks, then AI review of 10 risk categories |
| Recommendations | Yes | Renegotiation, escrow, pricing or walk-away, grounded in computed figures |
| Ask the agreement | Yes (retrieval-grounded) | Answers only from retrieved passages, with citations |

The financial assumptions the agreement does not contain (the acquirer's own financials, for example) start as illustrative estimates, marked with `*` in the Assumptions form on the **Cash flow** tab. Enter real figures there and press **Update simulation**. The simulator is a planning model for comparing scenarios and sizing protections, not a full merger model.

"""
Project settings - the ONLY file you need to edit.

Choose where the AI runs:
    LLM_PROVIDER = "ollama"   local open models on your own GPU (e.g. an NVIDIA T4), no API key, no data leaves the machine
    LLM_PROVIDER = "gemini"   Google's Gemini API (needs GEMINI_API_KEY)

End users of the app are never asked for a key or a model; everything is read from here.
Keep this file private - do not commit a real key to a public repository.
"""

LLM_PROVIDER = "ollama"

# ----------------------------------------------------------------------------- Ollama (local)
# Where the Ollama server runs. Same machine: leave as is.
OLLAMA_HOST = "http://localhost:11434"

# Chat model. Tested presets for a 16 GB NVIDIA T4 (pull them first: `ollama pull <name>`):
#   "qwen3:14b"   best quality that fits fully on a T4 (~9.3 GB)            <- default
#   "qwen3:8b"    about twice as fast, slightly weaker on long clauses (~5.2 GB)
OLLAMA_MODEL = "qwen3:14b"

# Embedding model for document search (small, runs on the same GPU).
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Context window per request, in tokens. 16384 is the sweet spot on a T4 with qwen3:14b.
# Use 24576 with qwen3:8b for longer excerpts; lower it to 12288 if you see out-of-memory errors.
OLLAMA_CONTEXT = 16384

# Requests processed at the same time. Must match OLLAMA_NUM_PARALLEL on the server (2 on a T4).
OLLAMA_PARALLEL = 2

# ----------------------------------------------------------------------------- Gemini (API)
# Your Gemini API key (starts with "AIza..."). Get one at https://aistudio.google.com/apikey
GEMINI_API_KEY = ""

# Gemini model used for every analysis step.
GEMINI_MODEL = "gemini-3.8-flash"

# Reasoning depth: "" (model default), "low" (fastest), "medium" or "high" (most thorough).
GEMINI_THINKING_LEVEL = ""

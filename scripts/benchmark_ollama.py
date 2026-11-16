"""Check that the local model runs well on this GPU.

    python3 scripts/benchmark_ollama.py          # speed + GPU placement + one structured call
    python3 scripts/benchmark_ollama.py --full   # also times a complete review of the sample agreement

Reports prompt-processing and generation speed (tokens/second), whether the model sits
100% in GPU memory (anything less means CPU offload and a big slowdown), and how long
one structured extraction takes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ma_engine.config import get_settings  # noqa: E402
from ma_engine.ingestion.loader import load_file  # noqa: E402
from ma_engine.llm import OllamaClient  # noqa: E402
from ma_engine.llm.connection import check_ollama  # noqa: E402
from ma_engine.models import QAAnswer  # noqa: E402
from ma_engine.pipeline import MAEngine  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_merger_agreement.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Time a complete review of the sample agreement")
    args = ap.parse_args()
    s = get_settings()
    cfg = s.llm
    if not cfg.is_local:
        print('utils.py has LLM_PROVIDER = "gemini". Set it to "ollama" to benchmark the local model.')
        return 1
    host = cfg.ollama_host.rstrip("/")
    status = check_ollama(cfg)
    print(status.message, status.detail)
    if not status.ok:
        return 1

    text = SAMPLE.read_text(encoding="utf-8")[:24_000]  # ~6k tokens of real contract text
    body = {"model": cfg.ollama_model, "stream": False, "think": False, "keep_alive": cfg.ollama_keep_alive,
            "messages": [{"role": "user", "content": text + "\n\nSummarise the termination provisions in 150 words."}],
            "options": {"num_ctx": cfg.num_ctx, "num_predict": 300, "temperature": 0.1}}
    with httpx.Client(timeout=900) as http:
        http.post(f"{host}/api/chat", json={**body, "options": {**body["options"], "num_predict": 1}})  # load model
        t = time.time()
        r = http.post(f"{host}/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text:
            body.pop("think")
            r = http.post(f"{host}/api/chat", json=body)
        r.raise_for_status()
        d = r.json()
        wall = time.time() - t
        pe = d.get("prompt_eval_count", 0) / max(d.get("prompt_eval_duration", 1) / 1e9, 1e-9)
        ge = d.get("eval_count", 0) / max(d.get("eval_duration", 1) / 1e9, 1e-9)
        print(f"\nPrompt processing: {pe:,.0f} tokens/s ({d.get('prompt_eval_count', 0):,} tokens)")
        print(f"Generation:        {ge:,.1f} tokens/s ({d.get('eval_count', 0)} tokens), {wall:.1f}s wall time")

        ps = http.get(f"{host}/api/ps").json().get("models", [])
        for m in ps:
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            share = 100 * vram / size if size else 0
            flag = "OK" if share >= 99 else "WARNING: part of the model is on the CPU - lower OLLAMA_CONTEXT or use a smaller model"
            print(f"{m.get('name')}: {size / 1e9:.1f} GB, {share:.0f}% on GPU  {flag}")

    client = OllamaClient(cfg)
    t = time.time()
    ans = client.structured(system="You are a precise M&A lawyer.", prompt="What fee does Parent pay and when?",
                            context=f'<excerpt id="s-c0001" section="">\n{text[-9000:]}\n</excerpt>',
                            output_model=QAAnswer, task="benchmark")
    print(f"\nStructured answer in {time.time() - t:.1f}s: {ans.answer[:200]}")

    if args.full:
        engine = MAEngine(client, s)
        t = time.time()
        ws = engine.build_workspace([load_file(SAMPLE)])
        report = engine.analyze(ws, progress=lambda m, f: print(f"  [{f:4.0%}] {m}  ({time.time() - t:.0f}s)"))
        print(f"\nFull review: {time.time() - t:.0f}s, score {report.scorecard.total}/100, {len(report.flags)} flags, "
              f"{report.usage.calls} model calls")
        for w in report.warnings:
            print("  note:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

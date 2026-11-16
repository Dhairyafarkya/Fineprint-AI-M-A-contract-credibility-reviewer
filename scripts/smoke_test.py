"""End-to-end smoke test on the sample agreement.

    python scripts/smoke_test.py          # offline, mock model (no key needed)
    python scripts/smoke_test.py --live   # real Gemini calls (key from utils.py)

Prints a compact report and writes it to smoke_test_report.md. With --live this
makes roughly 13-15 Gemini calls on the ~3,000-word sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ma_engine.config import get_settings  # noqa: E402
from ma_engine.ingestion.loader import load_file  # noqa: E402
from ma_engine.llm import GeminiClient, MockLLMClient  # noqa: E402
from ma_engine.pipeline import MAEngine  # noqa: E402
from ma_engine.reporting.markdown_report import report_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Call the real Gemini API")
    ap.add_argument("--file", type=Path, default=ROOT / "sample_data" / "sample_merger_agreement.txt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    llm = GeminiClient(settings.llm) if args.live else MockLLMClient()
    engine = MAEngine(llm, settings)
    t0 = time.time()
    ws = engine.build_workspace([load_file(args.file)])
    report = engine.analyze(ws, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    answer, _ = engine.ask(ws, "What happens to the target's debt at closing?")
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"Score: {report.scorecard.total}/100 ({report.scorecard.rating})")
    print(f"Flags: {len(report.flags)}  (" + ", ".join(
        f"{s}={sum(f.severity == s for f in report.flags)}" for s in ("critical", "high", "medium", "low")) + ")")
    if report.summary:
        print(f"Summary: {report.summary.headline}")
    if report.advice:
        print(f"Stance: {report.advice.overall_stance} - {len(report.advice.recommendations)} recommendations")
    print(f"Q&A: {answer.answer[:300]}")
    u = report.usage
    print(f"Usage: {u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out tokens, "
          f"{u.cache_read_tokens:,} cache-read; {elapsed:.1f}s")
    for w in report.warnings:
        print("WARNING:", w)
    out = ROOT / "smoke_test_report.md"
    out.write_text(report_to_markdown(report), encoding="utf-8")
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

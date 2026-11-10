"""Run the API for a demo recording.

Everything is the real stack — Docling parsing, chunking, fastembed
embeddings, hybrid search in Qdrant, cross-encoder reranking — except
answer generation, which is scripted so a demo can be recorded without
an LLM key. The scripted answer still cites the chunks retrieval
actually returned. Pass --live to use the configured Claude model instead.

    uv run python scripts/demo/demo_server.py --port 8765 [--live]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core.interfaces import Answer, Chunk, Citation  # noqa: E402

DEMO_MODEL = os.environ.get("DEMO_MODEL_LABEL", "claude-opus-5")
# Real generation takes a moment; keep the "thinking" state visible on video.
DEMO_DELAY = float(os.environ.get("DEMO_ANSWER_DELAY", "1.8"))


class ScriptedAnswerer:
    """Deterministic answers for the demo manual; cites real retrieved chunks."""

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        time.sleep(DEMO_DELAY)

        def find(*needles: str) -> int | None:
            for i, chunk in enumerate(chunks):
                if all(n.lower() in chunk.text.lower() for n in needles):
                    return i
            return None

        q = question.lower()
        codes = find("E03", "Cliff sensor")
        care = find("HEPA")
        if "e03" in q and codes is not None:
            cited = [codes] + ([care] if care is not None and care != codes else [])
            text = (
                "Error **E03** means the robot's **cliff sensor is blocked** [1]. "
                "To fix it, wipe the sensors with a dry cloth, then start the run again [1]."
            )
            if len(cited) > 1:
                text += (
                    "\n\nTo prevent it coming back, follow the routine maintenance: empty the "
                    "dustbin after every run and wash the HEPA filter monthly [2]."
                )
        elif chunks:
            cited = [0]
            text = "Here is what the manual says on this topic [1]."
        else:
            return Answer("No indexed document covers this question.", [], DEMO_MODEL)

        text = text.replace("**", "")
        citations = [Citation(marker=n, chunk=chunks[i]) for n, i in enumerate(cited, start=1)]
        return Answer(text=text, citations=citations, model=DEMO_MODEL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true", help="use the real Claude answerer")
    args = parser.parse_args()

    import uvicorn

    from app import deps
    from app.main import app
    from rag_core.config import get_settings
    from rag_core.pipeline import build_pipeline

    pipeline = build_pipeline(get_settings())
    if not args.live:
        pipeline.answerer = ScriptedAnswerer()
    deps._pipeline = pipeline  # the app's lazily-built singleton

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

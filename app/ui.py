"""Dashboard: server-rendered HTML fragments driven by HTMX.

The page shell at "/" is public; every fragment endpoint requires the
same X-API-Key as the JSON API (the browser stores the key locally and
HTMX attaches it to each request). No JavaScript build step — the
server renders HTML, HTMX swaps it into the page.

Hidden from the OpenAPI schema: Swagger documents the JSON API, and
these fragments are an implementation detail of the dashboard.
"""

from __future__ import annotations

import re
from pathlib import Path

import anthropic
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app.deps import get_pipeline, require_api_key, save_and_ingest
from rag_core.config import get_settings
from rag_core.llm import LLMNotConfiguredError

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def cite_markup(text: str) -> Markup:
    """Escape answer text, split paragraphs, and turn [n] markers into
    badges that link to the matching source card."""
    paragraphs = [p.strip() for p in str(text).split("\n\n") if p.strip()]
    html = []
    for paragraph in paragraphs:
        escaped = str(escape(paragraph)).replace("\n", "<br>")
        linked = _MARKER_RE.sub(r'<a class="cite" href="#src-\1">\1</a>', escaped)
        html.append(f"<p>{linked}</p>")
    return Markup("".join(html))


templates.env.filters["cite_markup"] = cite_markup


@router.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/ui/documents", dependencies=[Depends(require_api_key)])
def documents_fragment(request: Request, pipeline=Depends(get_pipeline)):
    return templates.TemplateResponse(
        request, "fragments/documents.html", {"documents": pipeline.list_sources()}
    )


@router.post("/ui/upload", dependencies=[Depends(require_api_key)])
def upload_fragment(request: Request, file: UploadFile, pipeline=Depends(get_pipeline)):
    try:
        result = save_and_ingest(file, pipeline)
    except HTTPException as exc:
        return _error(request, exc.detail)
    return templates.TemplateResponse(
        request,
        "fragments/documents.html",
        {"documents": pipeline.list_sources(), "just_indexed": result},
    )


@router.delete("/ui/documents/{source}", dependencies=[Depends(require_api_key)])
def delete_fragment(request: Request, source: str, pipeline=Depends(get_pipeline)):
    pipeline.delete_source(source)
    stored = Path(get_settings().uploads_dir) / Path(source).name
    stored.unlink(missing_ok=True)
    return templates.TemplateResponse(
        request, "fragments/documents.html", {"documents": pipeline.list_sources()}
    )


@router.post("/ui/query", dependencies=[Depends(require_api_key)])
def query_fragment(
    request: Request, pipeline=Depends(get_pipeline), question: str = Form("")
):
    if len(question.strip()) < 3:
        return _error(request, "Please type a question first.")
    try:
        answer = pipeline.query(question)
    except LLMNotConfiguredError as exc:
        return _error(request, str(exc))
    except anthropic.AuthenticationError:
        return _error(request, "The Anthropic API key is invalid.")
    except anthropic.APIConnectionError:
        return _error(request, "Could not reach the Anthropic API.")
    except anthropic.APIStatusError as exc:
        return _error(request, f"Upstream model error ({exc.status_code}).")
    return templates.TemplateResponse(request, "fragments/answer.html", {"answer": answer})


def _error(request: Request, message: str):
    return templates.TemplateResponse(request, "fragments/error.html", {"message": message})

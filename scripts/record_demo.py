"""Record the README / LinkedIn demo video: upload a PDF -> ask -> cited answer.

One command, run from the repo root:

    uv run --with playwright --with imageio-ffmpeg python scripts/record_demo.py

First time only, install the browser Playwright drives:

    uv run --with playwright playwright install chromium

What it does:
  1. builds a 2-page sample PDF (a product manual with a table),
  2. starts the API on a free port with a throwaway vector store,
  3. warms the models up (first-run downloads stay out of the video),
  4. drives the dashboard in a headless browser with on-screen captions,
  5. writes docs/demo.mp4 (H.264, 1280x720, plays on LinkedIn/GitHub).

Needs ANTHROPIC_API_KEY in the environment or in .env for the answer step.
Retrieval runs on free local models; one recording costs a single Claude call.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_KEY = "demo-key"
QUESTION = "My robot vacuum shows error E03 — what does it mean and how do I fix it?"

MANUAL_HTML = """<html><body style="font-family:Helvetica,Arial;padding:40px;color:#222">
<h1>Atlas X200 Robot Vacuum — User Manual</h1>
<h2>First setup</h2><p>Charge the Atlas X200 for at least 6 hours before first use. Place the
charging dock against a wall with 0.5 m of clearance on each side. Download the Atlas Home app
and pair the vacuum over Bluetooth by holding the power button for 3 seconds.</p>
<h2>Battery and charging</h2><p>The X200 uses a 5200 mAh lithium-ion battery rated for 180
minutes of cleaning per charge. A full recharge takes 4.5 hours. The battery is
user-replaceable; replacement part number is BAT-X200-52.</p>
<h2>Maintenance</h2><p>Empty the dustbin after every run. Wash the HEPA filter monthly under
cold water and let it dry for 24 hours before reinstalling. Replace the side brushes every
6 months and the main roller every 12 months.</p>
<h2 style="page-break-before:always">Error codes</h2>
<table border="1" cellpadding="8" style="border-collapse:collapse;width:100%">
<tr><th>Code</th><th>Meaning</th><th>Fix</th></tr>
<tr><td>E01</td><td>Wheel stuck</td><td>Remove debris from the wheel module</td></tr>
<tr><td>E02</td><td>Dustbin missing</td><td>Reinsert the dustbin until it clicks</td></tr>
<tr><td>E03</td><td>Cliff sensor blocked</td><td>Wipe the sensors with a dry cloth</td></tr>
<tr><td>E04</td><td>Battery too low to start</td><td>Dock the vacuum for 30 minutes</td></tr>
<tr><td>E05</td><td>Wi-Fi connection lost</td><td>Restart the router and re-pair</td></tr>
</table>
<h2>Storage</h2><p>For storage longer than one month, charge the battery to 60 percent,
power off the unit, and keep it below 35 °C.</p>
</body></html>"""

# Injected into every page: a caption bar and a click highlight, since
# headless recordings show no mouse cursor.
OVERLAY_JS = """
window.caption = (text) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div');
    el.id = '__cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);' +
      'z-index:99999;background:rgba(17,24,39,.92);color:#fff;' +
      'font:600 22px system-ui,sans-serif;padding:12px 24px;border-radius:10px;' +
      'box-shadow:0 8px 24px rgba(0,0,0,.25);transition:opacity .3s;max-width:90%;' +
      'text-align:center';
    document.body.appendChild(el);
  }
  el.style.opacity = text ? '1' : '0';
  if (text) el.textContent = text;
};
window.highlight = (sel) => {
  const el = document.querySelector(sel);
  if (!el) return;
  el.style.transition = 'box-shadow .2s';
  el.style.boxShadow = '0 0 0 4px rgba(11,95,255,.45)';
  setTimeout(() => (el.style.boxShadow = ''), 700);
};
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return resp.status


def wait_for_health(base: str, server: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit("API server exited during startup — see its output above.")
        try:
            if http("GET", f"{base}/health") == 200:
                return
        except OSError:
            time.sleep(1)
    sys.exit("API server did not become healthy in time.")


def upload(base: str, pdf: Path) -> None:
    boundary = "----demo-boundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{pdf.name}\"\r\nContent-Type: application/pdf\r\n\r\n"
    ).encode() + pdf.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    http(
        "POST",
        f"{base}/documents",
        body,
        {"X-API-Key": API_KEY, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def record(base: str, pdf: Path, video_dir: Path, browser) -> Path:
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=str(video_dir),
        record_video_size={"width": 1280, "height": 720},
    )
    # Escape hatch for sandboxes where unpkg.com is unreachable.
    if htmx := os.environ.get("DEMO_HTMX_FILE"):
        context.route(
            "https://unpkg.com/htmx.org@2.0.4",
            lambda route: route.fulfill(path=htmx, content_type="application/javascript"),
        )
    context.add_init_script(OVERLAY_JS)
    page = context.new_page()

    def cap(text: str) -> None:
        page.evaluate("t => window.caption(t)", text)

    def highlight(selector: str) -> None:
        page.evaluate("s => window.highlight(s)", selector)

    def scroll_to_bottom() -> None:
        page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")

    page.goto(base)
    page.add_style_tag(content="html{zoom:1.15}")
    cap("RAG Microservice — ask questions over your documents, get cited answers")
    page.wait_for_timeout(3000)

    cap("Authenticate with the service API key")
    highlight("#api-key")
    page.type("#api-key", API_KEY, delay=90)
    page.dispatch_event("#api-key", "change")
    page.wait_for_timeout(1200)

    cap("Upload any document — here, a 2-page PDF manual with a table")
    page.set_input_files("input[type=file]", str(pdf))
    page.wait_for_timeout(1200)
    upload_button = 'form[hx-post="/ui/upload"] button'
    highlight(upload_button)
    page.click(upload_button)
    cap("Docling parses the layout → structure-aware chunks → hybrid index in Qdrant")
    page.wait_for_selector(".notice", timeout=300_000)
    page.wait_for_timeout(2500)

    cap("Ask a question in plain language")
    page.click("input[name=question]")
    page.type("input[name=question]", QUESTION, delay=45)
    page.wait_for_timeout(600)
    ask_button = 'form[hx-post="/ui/query"] button'
    highlight(ask_button)
    page.click(ask_button)
    cap("Hybrid search (dense + BM25) → cross-encoder rerank → Claude")
    page.wait_for_selector("#answer article, #answer .error", timeout=180_000)
    if page.locator("#answer .error").count():
        message = page.locator("#answer .error").inner_text()
        context.close()
        sys.exit(f"The dashboard showed an error instead of an answer: {message}")

    cap("A grounded answer, with inline citations")
    scroll_to_bottom()
    page.wait_for_timeout(5000)

    summary = page.locator(".citations summary").first
    if summary.count():
        cap("Every citation opens the exact source passage")
        summary.click()
        scroll_to_bottom()
        page.wait_for_timeout(5500)

    cap("FastAPI · Qdrant · Docling · Claude — REST API, Docker, Kubernetes, evals")
    page.wait_for_timeout(4000)

    video = page.video
    context.close()
    return Path(video.path())


def to_mp4(webm: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(webm),
            # Drop the blank first frames while the page loads.
            "-ss", "0.6",
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.mp4")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    work = Path(tempfile.mkdtemp(prefix="rag-demo-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "RAG_API_KEY": API_KEY,
        "RAG_INSTANCE_NAME": "demo",
        "RAG_QDRANT_URL": "",
        "RAG_QDRANT_PATH": str(work / "qdrant"),
        "RAG_UPLOADS_DIR": str(work / "uploads"),
    }

    print(f"Starting API on {base} (scratch data in {work})")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get("DEMO_CHROMIUM"))

            pdf = work / "atlas-x200-manual.pdf"
            page = browser.new_page()
            page.set_content(MANUAL_HTML)
            page.pdf(path=str(pdf), format="A4")
            page.close()

            wait_for_health(base, server)
            print("Warming up models (first run downloads them — can take a few minutes)…")
            upload(base, pdf)
            http("DELETE", f"{base}/documents/{pdf.name}", headers={"X-API-Key": API_KEY})

            print("Recording…")
            webm = record(base, pdf, work / "video", browser)
            browser.close()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        to_mp4(webm, args.out)
        print(f"Done: {args.out}")
    finally:
        server.terminate()
        server.wait(timeout=30)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

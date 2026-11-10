"""Record the demo video (1920x1080 MP4) for the README / LinkedIn.

    uv run --with playwright playwright install chromium          # first time only
    uv run --with playwright --with imageio-ffmpeg python scripts/demo/record.py

Scenes: title card -> install in a terminal (with the service API key
step) -> the dashboard (connect with the key, upload a PDF, ask, open
citations) -> the same over curl -> closing card.

The terminal scenes are scripted animations. The dashboard and the curl
JSON come from a real server running the real ingestion and retrieval
pipeline (see demo_server.py); only the answer text is scripted unless
you pass --live (needs ANTHROPIC_API_KEY).
"""

# ruff: noqa: E501  (terminal transcript lines read better unwrapped)
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SERVICE_KEY = "8f3a2c71e94b05d6a1c3e7f2"  # what `openssl rand -hex 12` prints in the video
QUESTION = "My robot vacuum shows error E03 — what does it mean and how do I fix it?"
CURL_QUESTION = "What does error E03 mean?"

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


# --------------------------------------------------------------------------
# server helpers
# --------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(method: str, url: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return resp.read()


def wait_for_health(base: str, server: subprocess.Popen, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit("Demo server exited during startup — see its output above.")
        try:
            request("GET", f"{base}/health")
            return
        except OSError:
            time.sleep(1)
    sys.exit("Demo server did not become healthy in time.")


def upload(base: str, pdf: Path) -> None:
    boundary = "----demo-boundary"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{pdf.name}"\r\nContent-Type: application/pdf\r\n\r\n'
    ).encode() + pdf.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request(
        "POST",
        f"{base}/documents",
        body,
        {"X-API-Key": SERVICE_KEY, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


# --------------------------------------------------------------------------
# terminal content
# --------------------------------------------------------------------------


def _j(value) -> str:
    """One JSON scalar as colored HTML."""
    if isinstance(value, str):
        return f'<span class="str">{html.escape(json.dumps(value, ensure_ascii=False))}</span>'
    if value is None:
        return '<span class="dim">null</span>'
    return f'<span class="num">{json.dumps(value)}</span>'


def query_lines(data: dict) -> list[str]:
    """The /query response as `jq` would print it, citations one per line."""
    prop = lambda k: f'<span class="prop">"{k}"</span>'  # noqa: E731
    lines = [
        "{",
        f'  {prop("answer")}: {_j(data["answer"])},',
        f'  {prop("model")}: {_j(data["model"])},',
        f'  {prop("citations")}: [',
    ]
    cites = data["citations"]
    for i, c in enumerate(cites):
        fields = ", ".join(f"{prop(k)}: {_j(c[k])}" for k in ("marker", "source", "page", "heading"))
        lines.append("    {" + fields + "}" + ("," if i < len(cites) - 1 else ""))
    lines += ["  ]", "}"]
    return lines


# --------------------------------------------------------------------------
# the recording
# --------------------------------------------------------------------------


def record(base: str, pdf: Path, query_json: dict, video_dir: Path, browser) -> Path:
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir=str(video_dir),
        record_video_size={"width": 1920, "height": 1080},
        color_scheme="dark",
    )
    if htmx := os.environ.get("DEMO_HTMX_FILE"):  # sandboxes without unpkg.com
        context.route(
            "https://unpkg.com/htmx.org@2.0.4",
            lambda route: route.fulfill(path=htmx, content_type="application/javascript"),
        )
    page = context.new_page()
    page.goto((HERE / "stage.html").as_uri())
    page.wait_for_function("document.fonts.ready.then(() => true)")

    def js(expr: str, *args):
        return page.evaluate(expr, list(args) if args else None)

    def wait(ms: int) -> None:
        page.wait_for_timeout(ms)

    def cap(text: str | None) -> None:
        js("a => stage.caption(a[0])", text)

    def steps(name: str | None) -> None:
        js("a => stage.steps(a[0])", name)

    def show(scene: str) -> None:
        js("a => stage.show(a[0])", scene)

    def type_cmd(cmd: str, speed: int = 34) -> None:
        js("a => stage.type(a[0], a[1])", cmd, speed)

    def out(lines: list[str], gap: int = 90) -> None:
        js("a => stage.out(a[0], a[1])", lines, gap)

    # ---------------- 1. title ----------------
    show("intro")
    wait(4200)

    # ---------------- 2. install ----------------
    js("() => stage.termTitle('~/code — zsh')")
    show("term")
    steps("install")
    cap("Install in under a minute")
    wait(900)
    type_cmd("git clone https://github.com/RawCooked/rag-microservice.git && cd rag-microservice")
    out([
        "<span class='dim'>Cloning into 'rag-microservice'...</span>",
        "<span class='dim'>Receiving objects: 100% (214/214), 812 KiB | 9.4 MiB/s, done.</span>",
        "",
    ])
    wait(500)

    steps("key")
    cap("You choose the service API key — every client must send it as <code>X-API-Key</code>")
    type_cmd("export RAG_API_KEY=$(openssl rand -hex 12)")
    type_cmd("echo $RAG_API_KEY")
    out([f"<span class='key'>{SERVICE_KEY}</span>", ""])
    wait(1600)
    cap("Plus your model provider key, used only for answer generation")
    type_cmd("export ANTHROPIC_API_KEY=sk-ant-api03-••••••••••••")
    out([""])
    wait(900)

    steps("install")
    cap("One command: nginx load balancer + 2 API replicas + Qdrant")
    type_cmd("docker compose up -d --build --scale api=2")
    out([
        "<span class='dim'>[+] Building 41.2s (14/14) FINISHED</span>",
        "<span class='dim'>[+] Running 6/6</span>",
        " <span class='ok'>✔</span> Network rag-microservice_default      <span class='ok'>Created</span>",
        " <span class='ok'>✔</span> Volume  rag-microservice_models       <span class='ok'>Created</span>",
        " <span class='ok'>✔</span> Container rag-microservice-qdrant-1   <span class='ok'>Started</span>",
        " <span class='ok'>✔</span> Container rag-microservice-api-1      <span class='ok'>Started</span>",
        " <span class='ok'>✔</span> Container rag-microservice-api-2      <span class='ok'>Started</span>",
        " <span class='ok'>✔</span> Container rag-microservice-nginx-1    <span class='ok'>Started</span>",
        "",
    ], gap=260)
    wait(400)
    cap("Requests are load-balanced across replicas")
    type_cmd("for i in 1 2; do curl -si localhost:8080/health | grep X-Instance; done", 26)
    out([
        "x-instance: <span class='key'>3f2c9a1b7e04</span>",
        "x-instance: <span class='key'>b81d07e4c2a9</span>",
    ], gap=350)
    js("() => stage.prompt()")
    wait(2000)

    # ---------------- 3. dashboard: connect ----------------
    frame_el = page.locator("#app")
    js(f"() => document.getElementById('app').src = '{base}/'")
    frame = page.frame_locator("#app")
    frame.locator("#api-key").wait_for()
    app_frame = next(f for f in page.frames if f.url.startswith(base))
    app_frame.add_style_tag(content="html{zoom:1.5}")
    frame_el.wait_for()
    cap(None)
    show("browser")
    steps("key")
    wait(1200)

    def point(selector: str, ms: int = 750) -> tuple[float, float]:
        box = frame.locator(selector).first.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        js("a => stage.cursorTo(a[0], a[1], a[2])", x, y, ms)
        return x, y

    def click(selector: str) -> None:
        x, y = point(selector)
        js("a => stage.click(a[0], a[1])", x, y)
        frame.locator(selector).first.click()

    cap("Open the dashboard and paste the same key")
    click("#api-key")
    frame.locator("#api-key").press_sequentially(SERVICE_KEY, delay=55)
    click("#toggle-key")
    frame.locator("#api-key").dispatch_event("change")
    frame.locator("#status[data-state=ok]").wait_for()
    cap("Key accepted — <span style='color:#34d399'>● Connected</span> &nbsp;<small>wrong key = 401</small>")
    wait(2200)

    # ---------------- 4. upload ----------------
    steps("upload")
    cap("Upload any document — here, a 2-page PDF manual with a table")
    x, y = point("#dropzone")
    js("a => stage.click(a[0], a[1])", x, y)
    frame.locator("#file-input").set_input_files(str(pdf))
    wait(1100)
    click("#upload-btn")
    cap("Docling parses layout &amp; tables → structure-aware chunks → hybrid index in Qdrant")
    frame.locator(".notice").wait_for(timeout=300_000)
    wait(2600)

    # ---------------- 5. ask ----------------
    steps("ask")
    cap("Ask a question in plain language")
    click("input[name=question]")
    frame.locator("input[name=question]").press_sequentially(QUESTION, delay=38)
    wait(400)
    click(".ask-form button")
    cap("Hybrid search (dense + BM25) → cross-encoder rerank → grounded answer")
    frame.locator("#answer .answer, #answer .error").first.wait_for(timeout=180_000)
    if frame.locator("#answer .error").count():
        sys.exit("The dashboard showed an error: " + frame.locator("#answer .error").inner_text())
    app_frame.evaluate(
        "document.querySelector('#answer').scrollIntoView({behavior: 'smooth', block: 'start'})"
    )
    wait(900)
    cap("Every claim cites its source")
    point(".answer-body .cite", 650)
    wait(2600)

    cap("Open a citation to read the exact passage it came from")
    click(".source summary")
    wait(400)
    app_frame.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
    wait(4200)
    js("() => stage.cursorHide()")

    # ---------------- 6. REST API ----------------
    js("() => { stage.termClear(); stage.termTitle('~/code/rag-microservice — zsh'); }")
    show("term")
    steps("api")
    cap("Same service over REST — no key, no access")
    wait(700)
    type_cmd("curl -s localhost:8080/query -d '{\"question\": \"...\"}'", 30)
    out(["<span class='err'>{\"detail\": \"Missing or invalid X-API-Key header\"}</span>", ""])
    wait(1300)
    cap("With the key: answer + structured citations (file, page, heading)")
    type_cmd(
        'curl -s localhost:8080/query -H "X-API-Key: $RAG_API_KEY" '
        f"-d '{{\"question\": \"{CURL_QUESTION}\"}}' | jq",
        24,
    )
    out(query_lines(query_json), gap=70)
    js("() => stage.prompt()")
    wait(3800)

    # ---------------- 7. outro ----------------
    cap(None)
    steps(None)
    show("outro")
    wait(5000)

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
            "-ss", "0.5",  # skip the blank frame before the stage paints
            "-r", "30", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the demo video.")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.mp4")
    parser.add_argument("--live", action="store_true", help="real Claude answers")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    work = Path(tempfile.mkdtemp(prefix="rag-demo-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "RAG_API_KEY": SERVICE_KEY,
        "RAG_INSTANCE_NAME": "api-1",
        "RAG_QDRANT_URL": "",
        "RAG_QDRANT_PATH": str(work / "qdrant"),
        "RAG_UPLOADS_DIR": str(work / "uploads"),
    }
    cmd = [sys.executable, str(HERE / "demo_server.py"), "--port", str(port)]
    if args.live:
        cmd.append("--live")

    print(f"Starting demo server on {base} (first run downloads models — be patient)")
    server = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()

            pdf = work / "atlas-x200-manual.pdf"
            page = browser.new_page()
            page.set_content(MANUAL_HTML)
            page.pdf(path=str(pdf), format="A4")
            page.close()

            wait_for_health(base, server)
            print("Warming up the pipeline…")
            upload(base, pdf)
            query_json = json.loads(
                request(
                    "POST",
                    f"{base}/query",
                    json.dumps({"question": CURL_QUESTION}).encode(),
                    {"X-API-Key": SERVICE_KEY, "Content-Type": "application/json"},
                )
            )
            request("DELETE", f"{base}/documents/{pdf.name}", headers={"X-API-Key": SERVICE_KEY})

            print("Recording…")
            webm = record(base, pdf, query_json, work / "video", browser)
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

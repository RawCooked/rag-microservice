# API-key auth & the HTMX dashboard

## Auth
Every data endpoint requires an `X-API-Key` header, checked by one
FastAPI dependency ([deps.py](../../app/deps.py)). Declared through
`Security(APIKeyHeader(...))`, so Swagger UI grows an **Authorize**
button. `/health` stays public (load balancers and Kubernetes probes
must reach it without credentials), and so do `/docs` and the
dashboard shell.

This is the simplest real-world auth scheme. The next steps up —
per-user keys in a database, OAuth2/JWT — slot into the same
dependency without touching any endpoint.

## Dashboard (server-rendered + HTMX)
No React, no build step. The server renders HTML; **HTMX** attributes
on forms (`hx-post`, `hx-target`) submit them in the background and
swap the returned HTML *fragment* into the page. Upload, list, delete,
ask — each is a small endpoint returning a template.

The service API key is entered once in the browser, kept in
`localStorage`, and attached to every HTMX request as `X-API-Key` — the
dashboard authenticates exactly like any other API client, which keeps
one auth path for everything.

Why this stack in an AI project: it shows judgment. A single-page
framework here would be more code, more build tooling, and no more
capability — and the interview question "why HTMX?" has a crisp
answer: the server already owns the state; shipping it as HTML is the
shortest correct path.

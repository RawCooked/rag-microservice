# Grounded answers, citations, and honest errors

## Grounding
The model never answers from its own memory. It receives the question
plus the top retrieved passages numbered [1]..[n], with a system prompt
that says: answer ONLY from these, cite the numbers inline, and if the
passages don't contain the answer, say so. This is the main defense
against hallucination — the model summarizes evidence instead of
recalling "facts."

## Citations
The [n] markers in the response are parsed back into structured
citations: source file, page, heading, and the exact passage text.
Anyone can check any claim against its source — that's what makes the
service *verifiable* rather than "trust the AI."

Out-of-range markers (the model citing [9] when only 5 passages exist)
are dropped rather than shown — a small integrity detail.

## Refusals and fallbacks
Claude may decline a request for safety reasons (`stop_reason:
"refusal"`). The service handles that gracefully instead of crashing.
On models that support it (Claude Opus 5), we also enable server-side
fallbacks: a declined request is automatically retried on a fallback
model within the same API call.

## Honest HTTP errors
Each upstream failure mode maps to a truthful status code instead of a
generic 500 — most-specific first, the standard error-handling pattern:

| Situation | Status |
|---|---|
| No API key configured | 503 |
| Key invalid | 401 |
| Anthropic rate limit | 429 |
| Upstream/model error or network failure | 502 |

Clients (and the dashboard) can react correctly because the service
tells the truth about *whose* fault the failure was.

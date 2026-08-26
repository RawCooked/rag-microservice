# Architecture: why a core package + a thin API shell

```
rag_core/   ← all the RAG logic, zero web code, reusable anywhere
app/        ← FastAPI shell: HTTP, auth, dashboard. Thin on purpose.
```

**The rule:** `rag_core` never imports from `app`. The web layer is a
consumer of the core, like any other program would be (a CLI, a batch
job, someone else's service).

**Why it matters:**
- *Reusability* — `rag_core` can be pip-installed into another project.
- *Testability* — pipeline logic is tested without spinning up a server.
- *Swappability* — every stage is a Protocol in `interfaces.py`; a local
  embedding model and a cloud embedding API are interchangeable because
  the rest of the code only knows the contract, not the implementation.

This is hexagonal / ports-and-adapters architecture in miniature: the
interfaces are the ports, the concrete classes are the adapters.

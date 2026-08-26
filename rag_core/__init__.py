"""rag_core — reusable RAG pipeline components.

This package contains zero web code. It can be imported into any
project (a FastAPI app, a CLI, a batch job) and every component is
swappable via the protocols in `interfaces.py`.
"""

import os

# Windows blocks symlink creation unless Developer Mode is on, which
# breaks HuggingFace Hub's model cache (WinError 1314) when the local
# models download. Copying instead of symlinking is safe everywhere.
# setdefault: an explicit value in the real environment still wins.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

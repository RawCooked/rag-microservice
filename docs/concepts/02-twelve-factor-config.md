# 12-factor config: environment variables everywhere

All configuration — API keys, model names, database URLs — comes from
**environment variables**, never from hardcoded values.

- In dev: variables load from a `.env` file (git-ignored; `.env.example`
  documents every variable without real secrets).
- In Docker: `docker-compose.yml` sets them per container.
- In Kubernetes: ConfigMaps (plain config) and Secrets (keys) inject them.

**The payoff:** one build artifact (the Docker image) runs unchanged in
every environment. Deploying to a new cloud = setting new env vars, not
editing code. This is factor III of the "12-factor app" methodology that
most modern platform teams follow.

Concrete example in this repo: `RAG_QDRANT_URL` empty → the code uses an
embedded local vector store (no server needed). Set it to
`http://qdrant:6333` → the same code talks to a real Qdrant container.
Zero code changes between laptop and cluster.

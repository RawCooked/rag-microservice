# Docker, docker-compose, and nginx

## Containers
A container packages the app **plus its whole environment** — Python
version, libraries, system packages — into one runnable unit that
behaves identically on any machine with Docker. Unlike a virtual
machine it shares the host's kernel: starts in seconds, light on RAM.

- **Image** = the frozen build result (like a class).
- **Container** = a running instance of an image (like an object).
- **Dockerfile** = the recipe. Ours is **multi-stage**: a builder stage
  installs dependencies with uv, the final stage keeps only the
  virtualenv and code — smaller and safer. Dependencies are copied in
  *before* the code so the big dependency layer is cached between
  rebuilds.
- The container runs as a **non-root user** — if the app is ever
  compromised, the blast radius is smaller.

## docker-compose
One YAML file describing the whole system — nginx, N api replicas,
qdrant — started with a single command:

```
docker compose up --build --scale api=2
```

Named **volumes** persist what must survive container recreation: the
downloaded ML models, the vector index, uploaded files.

## Reverse proxy & load balancing (nginx)
nginx is the single public entrance (only it publishes a port; the API
containers are reachable solely on the internal network). It:

- **load-balances**: Docker DNS resolves `api` to every replica's IP
  and nginx round-robins across them. The API returns an `X-Instance`
  header naming the container that answered — run
  `curl -i localhost:8080/health` twice and watch it alternate.
- **rate-limits**: 10 req/s per client IP, protecting the expensive
  endpoints from abuse before they're even reached.
- absorbs slow uploads, sets forwarding headers, and would terminate
  HTTPS in production.

**Failover demo:** `docker stop <one api container>` — requests keep
succeeding through the remaining replica.

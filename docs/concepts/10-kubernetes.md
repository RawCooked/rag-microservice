# Kubernetes

docker-compose runs a system on **one machine**. Kubernetes runs it on
a **fleet** — and works declaratively: you state the desired end state
and the cluster continuously reconciles reality toward it. You don't
say "start two containers"; you say "two replicas must exist", and if
one dies, Kubernetes notices and replaces it.

## The objects used in `k8s/`

| Object | Role | In this project |
|---|---|---|
| **Pod** | Smallest unit: one running container (usually) | Created for you — never written by hand |
| **Deployment** | "Keep N identical pods alive; roll out updates gradually" | `rag-api` (2 replicas), `qdrant` (1) |
| **Service** | Stable internal address load-balancing across a Deployment's pods | `rag-api:8000`, `qdrant:6333` — replaces nginx's balancing role |
| **ConfigMap / Secret** | Config and credentials injected as env vars | same 12-factor vars as `.env` and compose |
| **PersistentVolumeClaim** | Storage that outlives pods | Qdrant's index |
| **Ingress** | Front door routing external HTTP to Services | optional here; `kubectl port-forward` is simpler locally |

## Probes
The Deployment declares two health checks against `/health`:
- **readiness** — pod gets no traffic until it answers (model loading
  takes a while on first start);
- **liveness** — pod is restarted if it stops answering.

## Try it (Docker Desktop has a built-in cluster)
```
docker build -t rag-microservice:latest -f docker/Dockerfile .
kubectl apply -f k8s/
kubectl get pods                      # 2× rag-api, 1× qdrant
kubectl port-forward svc/rag-api 8080:8000
kubectl delete pod <one rag-api pod>  # watch it come back
```

## Honest framing (worth repeating in interviews)
For a service this size, compose on one VM is the right production
choice; Kubernetes earns its complexity with many services, many
machines, or elastic scaling. Knowing *when not to use it* is the
senior answer.

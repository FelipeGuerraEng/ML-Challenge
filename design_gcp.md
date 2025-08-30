# design_gcp.md 

## Goal
Operate this mini‑prod churn project on Google Cloud with simple, cost‑aware components: reproducible training, containerized serving, basic drift/health monitoring, and CI/CD. 

## Reference architecture 
```
 Source control ──► CI/CD ──► Container registry ──► Serverless serving
                              │
                              └─► Batch training (on demand)
Data: object storage (artifacts) + warehouse (metrics/telemetry)
Monitoring: metrics + scheduled checks
```

## Build & CI/CD 
- **Container registry:** Artifact Registry for storing container images.
- **CI system:** GitHub Actions or Cloud Build to install dependencies, run tests, and push images.
- **Authentication from CI:** IAM Workload Identity Federation to avoid long‑lived keys.

## Training 
- **Runner:** Vertex AI Custom Training for containerized training jobs (alternatively Cloud Run Jobs for lightweight CPU jobs).
- **Data sources:** Cloud Storage for files; BigQuery for tabular data and historical metrics.
- **Outputs:** Model artifacts and metric summaries written back to Cloud Storage.
- **Scheduling:** Cloud Scheduler to trigger retraining; optionally orchestrate multi‑step flows with Cloud Workflows.

## Model management 
- **Versioned artifacts:** Store immutable versions in Cloud Storage with a lightweight “current” pointer.
- **Registry:** Vertex AI Model Registry to track versions, lineage, evaluations, and promotions.

## Serving
- **Serving platform:** Cloud Run for HTTP autoscaling to zero; deploy the FastAPI container.
- **Artifact resolution:** Fetch the latest model from Cloud Storage at startup to a local read‑only directory.
- **API management:** API Gateway for simple edge control or Apigee for full enterprise API management (security, quotas, analytics, dev portal).

## Monitoring & observability
- **Logs:** Cloud Logging for application logs and request traces emitted by the service.
- **Metrics:** Cloud Monitoring for built‑in and custom metrics (latency, throughput, quality signals).
- **Data drift:** Periodic drift checks comparing recent windows to a reference sample; persist summaries in Cloud Storage and/or BigQuery. Vertex AI Model Monitoring is an optional managed alternative for skew/drift detection.
- **Alerting:** Cloud Monitoring alerting policies driven by custom metrics or agent‑emitted status (e.g., `warn`/`critical`).

## Orchestration 
- **Scheduler:** Cloud Scheduler to trigger drift checks and the agent monitor on a cadence.
- **Workflow engine:** Cloud Workflows to coordinate multi‑step jobs (fetch data → compute → persist).

## Security & access (services)
- **Identity & access:** IAM service accounts with least‑privilege roles for reading storage/warehouse and invoking services.
- **Secrets:** Secret Manager for configuration and credentials; inject at runtime.

## Cost posture
- Favor serverless components (Cloud Run, Cloud Scheduler) that scale to zero for idle workloads.
- Keep training CPU‑bound unless profiling justifies accelerators; cap parallelism/trials in Vertex AI.
- Centralize artifacts and telemetry in managed storage/warehouse to simplify lifecycle and reduce ops overhead.

## Rollout & versioning
- Treat images as immutable; tag with commit/sha and environment labels in Artifact Registry.
- Promote by updating a “current” artifact pointer in Cloud Storage or promoting a version in Vertex AI Model Registry.
- Keep the prior model available to allow quick rollback if the agent marks `critical`.

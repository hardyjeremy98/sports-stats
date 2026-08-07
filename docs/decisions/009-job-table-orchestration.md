# ADR 009: Orchestration lives in the job table; cloud services supply workers, never logic

**Status:** Accepted
**Date:** 2026-08-08

## Context

The pipeline is moving toward parallel processing of full matches — frame-sharded feature
generation and chunked tracking, targeting roughly 16–32 GPU workers per half and ~30–45 min per
match (working analysis: Notion → 🛠️ Engineering → "Full-match processing — split, latency floor,
cost", curated 2026-08-07) — and, at the app repo's AWS phase ("P3" in `monorepo/docs/prd.md` §5),
deployment to AWS. Both create pressure to adopt a cloud orchestrator: AWS Batch `dependsOn`,
Step Functions, or similar services model fan-out, barriers and retries natively.

The existing architecture already contains an orchestrator: `matchlab_server`'s job table plus a
polling worker (`worker.py`), designed so that *any box that reaches the DB and the data volume
can be a worker*. This ADR elaborates locked decision #6 (cited in `models.py` and `worker.py`:
job table + async worker, no cloud vendor SDK); it does not change it.

Adopting a cloud orchestrator would split orchestration across two systems: phases and
dependencies expressed in a vendor's state language in production, and something else — or
nothing — locally. The local environment would stop being a faithful test of the production
system, and the worker model's deliberate vendor neutrality would be spent.

## Decision

All orchestration logic — phases, shard ranges, chunk dependencies, barriers, claims, and
workflow-level retry — lives in the job table and is executed by `matchlab_server` code. This is
the single orchestrator, local and deployed.

Cloud services are confined to two narrow roles, behind two seams:

1. **Worker supply.** A cloud compute service (e.g. AWS Batch with a scale-to-zero GPU compute
   environment) runs containers whose entrypoint is the same polling worker used locally. Scaling
   workers up and down is permitted (a scale-up trigger may *read* queue depth from the job
   table — read-only observation is not orchestration); deciding what a worker does next is not.
2. **Storage.** Run artifacts live behind `ArtifactStore`. It is `Path`-returning today — plain
   files are a documented design commitment (portable, diffable, directly servable to the Lab
   UI) — so the near-term deployed backend must be POSIX-mountable (e.g. EFS). An S3-backed
   object store would change the `ArtifactStore` API and is a separate decision, not made here.

**The boundary test:** if a cloud service must read or write job semantics — phase, shard,
dependency, outcome, retry count — to do its work, it is orchestration and is forbidden. If it
only decides how many containers exist and on what hardware, it is worker supply. Consequently,
Batch `retryStrategy` is set to 1 attempt, and recovery from a spot interruption is a job-table
lease expiry, never a Batch retry.

Parallel-processing support (parent/phase/chunk columns, a dependency-aware claim query) is
implemented in the job table; its reference implementation is N worker processes on one machine.

## What this requires (not yet built)

- **A lease/heartbeat column and a reaper.** Today a killed worker leaves its job `RUNNING`
  forever; `claim_next_job` only considers `QUEUED`, and `attempts` is incremented but never
  consulted. Workflow-level retry is this ADR's responsibility and must be implemented in
  `matchlab_server`, not delegated to the cloud service.
- **Schema growth via `db.py::_micro_migrations`** as today — noting that dependency-aware
  claiming will want composite indexes, which the additive-ALTER helper is not currently shaped
  for.

## Consequences

- Local development tests the real orchestrator — with one caveat that matters: claim safety
  (`SKIP LOCKED`) is only real on Postgres. The default SQLite dev DB ignores the hint, so
  multi-worker claim races must be exercised against the docker-compose Postgres, not `make dev`.
- The AWS deployment layer reduces to configuration: an image, a compute environment, a storage
  mount, a scale-up trigger. No application logic exists only in the cloud.
- No AWS SDK enters `matchlab_core`, `matchlab_server`, or `matchlab_train`. Cloud-specific code
  is confined to infrastructure-as-code (and a storage adapter if one is ever decided), mirroring
  the app repo's "no AWS SDK before the AWS phase" rule.
- Using cloud features for worker-pool mechanics (instance selection, spot lifecycle
  notification) is fine wherever it passes the boundary test above.

## Alternatives rejected

- **AWS Batch array jobs + `dependsOn` as the orchestrator.** Models the fan-out/barrier shape
  well, but phase logic would exist only in Batch job graphs — untestable locally, and retry
  authority would split between `Job.attempts` and Batch `retryStrategy`.
- **Step Functions (Distributed Map).** The best native fit for map→reduce→map→reduce, and the
  most vendor-specific option: orchestration state moves into ASL, directly spending the vendor
  neutrality that locked decision #6 bought.
- Either would be right if workflow scale outgrew a polling model — see below.

## Reconsider if

Workflow scale outgrows a polling model (thousands of concurrent shards where claim contention on
one DB becomes the bottleneck), or a managed orchestrator becomes necessary for operational
reasons. Even then, prefer generating the cloud workflow *from* the job table's model over
hand-maintaining two orchestration definitions.

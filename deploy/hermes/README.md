# AWG Hermes deployment

This overlay adds one native Hermes runtime per authenticated OpenWebUI user. The provisioner is the only service with the Docker socket. Agent containers have no socket, host bind mounts, published ports, or route to the public Internet. They share only the internal runtime network with OpenWebUI and the existing Qwen llama.cpp service.

## Immutable inputs

Build `deploy/hermes/Dockerfile` from the verified Hermes `v2026.9.14` source commit `345cd2b057a452236de401d3534b8502a7465e8d` and a base image pinned by digest. The release pipeline must inspect the result, then replace the null fields in `release-manifest.json` with the verified upstream and final runtime digests and a verification timestamp. Deployment fails closed while `runtime_image` is null or differs from `AWG_HERMES_RUNTIME_IMAGE`.

Run `build-release.sh <clean-hermes-checkout> <registry/upstream:release> <registry/runtime:release> <manifest-output>` from `deploy/hermes`. It rejects a checkout at any other commit or with local changes, builds and pushes the upstream image from that checkout, resolves its registry digest, then builds the AWG runtime from that digest. Both builds request BuildKit provenance and SBOM attestations. The script generates the only accepted manifest shape; the provisioner verifies release, commit, upstream digest, final digest, timestamp, and matching OCI labels before creating a runtime.

The `Publish immutable Hermes images` workflow runs after a relevant merge to `dev`, or by manual dispatch from `dev`. It always checks out Hermes at commit `345cd2b057a452236de401d3534b8502a7465e8d`; there is no workflow input for another upstream ref. It publishes the private packages `ghcr.io/mazazyrik/hermes-agent-upstream:<AWG merge SHA>` and `ghcr.io/mazazyrik/awg-hermes-runtime:<AWG merge SHA>`, verifies their immutable digests, attestations, runtime labels, and runtime user, then uploads `hermes-release-manifest-<AWG merge SHA>` as the deployment artifact. Download that artifact from the successful merge-commit run and deliver its `hermes-release-manifest.json` as described below. A manual run must be dispatched from the `dev` branch; other refs are rejected by the job condition.

Deliver exactly two repo-owned files to `/opt/awg-confluence-rag/deploy`: copy `docker-compose.hermes.yml` unchanged and copy the populated manifest as `hermes-release-manifest.json`. The verifier and compose overlay use these paths. Run `docker compose -f /opt/awg-confluence-rag/docker-compose.yml -f /opt/awg-confluence-rag/deploy/docker-compose.hermes.yml config --quiet` before rollout.

Supply `AWG_HERMES_CONTROL_SECRET` and `AWG_HERMES_QWEN_API_KEY` from the production secret store. They must never be committed or written to compose files. The control secret authenticates OpenWebUI to the provisioner. A persistent container is reused per user while a Redis lease serializes that user's runs. Each run receives a short-lived capability through runtime tmpfs; the token is absent from the container environment and persistent volumes. Temporary chats use disposable volumes and cannot persist memory.
Hermes also requires the shared OpenWebUI Redis connection. It fails closed without Redis; leases, active capabilities, revocation, and artifact handoff are never kept in process memory.

## Qwen capacity gate

The default command preserves two slots with `-c 131072 -np 2` and uses `q8_0` KV cache. Before enabling all users, run two simultaneous 64K-context sessions and one hour of mixed load. Record latency, quality, VRAM/RAM, OOMs, and output parity. If `q8_0` does not fit, test `q4_0` in an otherwise identical canary and approve it only after the 30-case corpus shows no critical regression and at least 90% success. If neither cache meets both capacity and quality gates, leave Hermes disabled.

## Release and rollback

1. Run backend regression tests, Hermes bridge tests, the 30-case corpus, two-user isolation checks, document round trips, cancellation, restart persistence, plugin rollback, and compose verification.
2. Merge only with no P0/P1 findings and green required checks. Build from the merge commit, publish immutable images, and record both digests.
3. Back up the OpenWebUI database/config and the Hermes named volumes. Deploy the overlay with closed test access and run `scripts/verify-awg-openwebui-deployment.sh hermes-compose` and `hermes-runtime`.
4. Switch the configured AWG model IDs only after the canary passes. Monitor agent failure rate, Confluence preflight traces, citation rejection, OOMs, cancellation latency, and cross-user denials.
5. Roll back by setting `AWG_HERMES_ENABLED=false`, restoring the previous immutable OpenWebUI image and llama command, and leaving Hermes volumes intact for investigation. Restore the database only if a migration or data corruption requires it.

The Docker socket on the provisioner remains a privileged residual risk. The service exposes only signed ensure/delete operations, validates fixed identifiers and endpoints, and creates containers from a fixed digest with a fixed security profile. Production should additionally constrain the provisioner with a Docker socket proxy or a dedicated node policy.

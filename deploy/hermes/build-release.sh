#!/bin/sh
set -eu

release='v2026.9.14'
commit='345cd2b057a452236de401d3534b8502a7465e8d'
source_checkout=${1:?Hermes source checkout is required}
upstream_ref=${2:?Upstream registry reference is required}
runtime_ref=${3:?Runtime registry reference is required}
manifest=${4:-release-manifest.generated.json}
package_source='https://github.com/mazazyrik/AWG-openwebui'

test "$(git -C "$source_checkout" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_checkout" status --porcelain)"
case "$upstream_ref" in *@*) exit 2 ;; esac

docker buildx build \
  --label "org.opencontainers.image.source=$package_source" \
  --label 'com.awg.hermes.upstream-source=https://github.com/NousResearch/hermes-agent' \
  --label "com.awg.hermes.upstream-revision=$commit" \
  --provenance=mode=max \
  --sbom=true \
  --push \
  --tag "$upstream_ref" \
  "$source_checkout"

upstream_digest=$(docker buildx imagetools inspect "$upstream_ref" --format '{{json .Manifest.Digest}}' | tr -d '"')
case "$upstream_digest" in sha256:*) ;; *) exit 2 ;; esac
upstream_repository=${upstream_ref%:*}
base_image="$upstream_repository@$upstream_digest"

docker buildx build \
  --file Dockerfile \
  --build-arg "HERMES_BASE_IMAGE=$base_image" \
  --build-arg "HERMES_RELEASE=$release" \
  --build-arg "HERMES_UPSTREAM_COMMIT=$commit" \
  --build-arg "AWG_PACKAGE_SOURCE=$package_source" \
  --provenance=mode=max \
  --sbom=true \
  --push \
  --tag "$runtime_ref" \
  .

runtime_digest=$(docker buildx imagetools inspect "$runtime_ref" --format '{{json .Manifest.Digest}}' | tr -d '"')
case "$runtime_digest" in sha256:*) ;; *) exit 2 ;; esac
runtime_repository=${runtime_ref%:*}
verified_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
jq -n \
  --arg release "$release" \
  --arg commit "$commit" \
  --arg base "$base_image" \
  --arg runtime "$runtime_repository@$runtime_digest" \
  --arg verified_at "$verified_at" \
  '{hermes_release:$release,hermes_upstream_commit:$commit,upstream_base_image:$base,runtime_image:$runtime,verified_at:$verified_at}' >"$manifest"

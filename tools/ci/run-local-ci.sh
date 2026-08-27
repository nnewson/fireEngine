#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/ci/run-local-ci.sh [format|configure|build|tidy|test|release-contract|all|shell]

Runs the GitHub Actions Linux checks inside Docker. The repository is copied into
the container before each run; Docker volumes hold the Linux build tree and vcpkg
checkout/cache so host build artifacts are left alone.

`release-contract` builds a genuine Release tree and runs the [release-contract]
cases — the suite's #ifdef NDEBUG bodies, which every other stage compiles away.
`all` includes it; the macOS runner deliberately does not (the contract is
platform-independent, so one job proves it).
EOF
}

mode="${1:-all}"
case "${mode}" in
  format|configure|build|tidy|test|release-contract|all|shell) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform="${DOCKER_PLATFORM:-linux/amd64}"
platform_tag="${platform//\//-}"
image="fireengine-local-ci:ubuntu-24.04-${platform_tag}"
build_volume="fireengine-local-ci-build-${platform_tag}"
# The Release-contract tree gets its own volume for the same reason the Dev one has one: without it
# every `release-contract` run recompiles all ~270 translation units inside an emulated container.
release_build_volume="fireengine-local-ci-build-release-${platform_tag}"
vcpkg_volume="fireengine-local-ci-vcpkg-${platform_tag}"
build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
docker_run_flags=(--rm)

if [ "${mode}" = "shell" ]; then
  docker_run_flags+=(--interactive)
  if [ -t 0 ]; then
    docker_run_flags+=(--tty)
  fi
fi

docker build \
  --platform "${platform}" \
  --tag "${image}" \
  --file "${repo_root}/tools/ci/Dockerfile" \
  "${repo_root}/tools/ci"

docker run "${docker_run_flags[@]}" \
  --platform "${platform}" \
  --volume "${repo_root}:/repo:ro" \
  --volume "${build_volume}:/work/fireEngine/build" \
  --volume "${release_build_volume}:/work/fireEngine/build-release" \
  --volume "${vcpkg_volume}:/cache/vcpkg" \
  --env VCPKG_ROOT=/cache/vcpkg \
  --env VCPKG_DISABLE_METRICS=1 \
  --env CI_RELEASE_CONTRACT=1 \
  --env CMAKE_BUILD_PARALLEL_LEVEL="${build_parallel_level}" \
  --workdir /work/fireEngine \
  "${image}" \
  fireengine-ci "${mode}"

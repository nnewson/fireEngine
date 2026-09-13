#!/usr/bin/env bash
set -euo pipefail

# Entry point baked into the Linux CI image (see Dockerfile). Reproduces the GitHub Actions Linux
# checks inside the container. Docker-specific bits (source sync, fresh vcpkg) live here; the actual
# build/test/lint stages come from the shared ci-stages.sh so this and the native macOS runner stay
# in lockstep.

mode="${1:-all}"

sync_source()
{
    # Every build tree is excluded, and for two reasons at once: a host CMakeCache names host
    # ABSOLUTE paths, so copying one in makes the container's configure fail outright ("the source
    # /work/fireEngine/CMakeLists.txt does not match the source /Users/... used to generate
    # cache"); and each of these is a mounted volume, which `--delete` would otherwise empty on
    # every run.
    rsync -a --delete \
        --exclude /.git \
        --exclude /build \
        --exclude /build-release \
        --exclude /vcpkg \
        --exclude /vcpkg_installed \
        /repo/ /work/fireEngine/
}

# The vcpkg TOOL commit, pinned to the same SHA as the manifest baseline in
# vcpkg-configuration.json and as the GitHub jobs' checkout. The baseline pins port versions; this
# pins everything else vcpkg brings (the tool itself, the triplets, the toolchain scripts). A
# replica that tracked master would drift away from the CI it exists to reproduce. Move it with the
# baseline, never on its own.
VCPKG_COMMIT=a1cae005c39be7b18ba319fced856b68d7276271

ensure_vcpkg()
{
    if [ ! -d "${VCPKG_ROOT}/.git" ]; then
        rm -rf "${VCPKG_ROOT:?}"/*
        git clone https://github.com/microsoft/vcpkg "${VCPKG_ROOT}"
    fi
    # Fetched BY SHA rather than by branch: the volume persists between runs, so a checkout already
    # holding the pinned commit does no network work at all, and one holding an older pin moves to
    # exactly this commit rather than to whatever master is today.
    if ! git -C "${VCPKG_ROOT}" cat-file -e "${VCPKG_COMMIT}^{commit}" 2>/dev/null; then
        git -C "${VCPKG_ROOT}" fetch --depth=1 origin "${VCPKG_COMMIT}"
    fi
    git -C "${VCPKG_ROOT}" checkout --detach "${VCPKG_COMMIT}"
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
}

# The Linux container installs the versioned clang-format from apt.llvm.org.
export CLANG_FORMAT=clang-format-22

# Shared stage bodies (copied next to this script in the image; sibling in the repo tree).
# shellcheck source=ci-stages.sh
. "$(dirname "${BASH_SOURCE[0]}")/ci-stages.sh"

sync_source
ci_print_versions

if [ "${mode}" = "shell" ]; then
    exec bash
fi

if [ "${mode}" != "format" ]; then
    ensure_vcpkg
fi

ci_run_stage "${mode}"

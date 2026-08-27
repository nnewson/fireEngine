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

ensure_vcpkg()
{
    if [ ! -d "${VCPKG_ROOT}/.git" ]; then
        rm -rf "${VCPKG_ROOT:?}"/*
        git clone https://github.com/microsoft/vcpkg "${VCPKG_ROOT}"
    else
        git -C "${VCPKG_ROOT}" fetch --depth=1 origin master
        git -C "${VCPKG_ROOT}" checkout --detach FETCH_HEAD
    fi
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

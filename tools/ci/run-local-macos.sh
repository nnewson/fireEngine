#!/usr/bin/env bash
set -euo pipefail

# Native macOS CI-parity runner. macOS *is* a target platform, so — unlike run-local-ci.sh, which
# spins up a Linux container — this runs the same stages directly on the host toolchain. It assumes
# the working local macOS dev setup you already build fireEngine with: Apple Clang, a Vulkan SDK /
# MoltenVK, glfw, a shader compiler (glslc or glslangValidator), and vcpkg via VCPKG_ROOT. It does
# NOT install anything.
#
# Stages mirror run-local-ci.sh: format | configure | build | tidy | test | all. The one exception
# is `release-contract`, which is Linux-only ON PURPOSE — those cases assert what a writer returns
# once NDEBUG removes its assertion, which does not vary by platform, so a second job would spend
# runner minutes to re-prove it. `all` here therefore stays "every gate macOS owns".
#
# Note: clang-tidy on macOS uses Apple's clang-tidy, which can differ from the Linux one the CI /
# Docker replica runs — treat the macos `tidy` stage as advisory; Linux is the source of truth.
# Note: configure uses -DFIRE_ENGINE_WARNINGS_AS_ERRORS=ON in the standard `build/` dir (CI parity),
# so it stays ON for later plain builds — reconfigure with =OFF to relax.

usage()
{
    cat <<'EOF'
Usage: tools/ci/run-local-macos.sh [format|configure|build|tidy|test|all]

Runs the CI stages natively on macOS (no Docker). Requires VCPKG_ROOT set to your vcpkg checkout
and a working local Vulkan/MoltenVK + glfw + shader-compiler toolchain — the setup you already
build fireEngine with.
EOF
}

mode="${1:-all}"
case "${mode}" in
    format | configure | build | tidy | test | all) ;;
    -h | --help | help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

if [ "${mode}" != "format" ] && [ -z "${VCPKG_ROOT:-}" ]; then
    echo "VCPKG_ROOT is not set — point it at your vcpkg checkout (the vcpkg preset's toolchain needs it)." >&2
    exit 2
fi

# macOS uses the unversioned clang-format; warn if it isn't the major version CI pins (22).
export CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
if command -v "${CLANG_FORMAT}" >/dev/null 2>&1; then
    fmt_major="$("${CLANG_FORMAT}" --version | sed -nE 's/.*version ([0-9]+).*/\1/p')"
    if [ "${fmt_major}" != "22" ]; then
        echo "warning: ${CLANG_FORMAT} is major ${fmt_major:-unknown}, but CI pins clang-format-22 — formatting may disagree." >&2
    fi
fi

# shellcheck source=ci-stages.sh
. "$(dirname "${BASH_SOURCE[0]}")/ci-stages.sh"

ci_print_versions
ci_run_stage "${mode}"

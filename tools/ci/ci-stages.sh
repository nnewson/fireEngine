#!/usr/bin/env bash
# Shared CI stage bodies. Sourced (never executed directly) by:
#   - container-run.sh    — the Linux/Docker replica (tools/ci/run-local-ci.sh)
#   - run-local-macos.sh  — the native macOS replica
#   - (mirrored by .github/workflows/ci.yml, which runs the same commands)
# Keeping the stage commands in one place stops the Docker, native, and GitHub paths drifting apart.
#
# Tunables via the environment:
#   CLANG_FORMAT          clang-format binary (clang-format-22 in the container; clang-format on macOS)
#   WARNINGS_AS_ERRORS    ON/OFF for the configure step (default ON — CI parity)
#   CI_RELEASE_CONTRACT   1 to include the Release-contract stage in `all` (Linux only — see below)

: "${CLANG_FORMAT:=clang-format}"
: "${WARNINGS_AS_ERRORS:=ON}"
# LINUX ONLY, and deliberately so. The `[release-contract]` cases assert what a writer returns once
# NDEBUG compiles its assertion away, which is platform-independent behaviour — one job proves it,
# and a second would only spend runner minutes. The Linux runner sets this; the macOS one leaves it
# at 0, so `all` means "every gate this platform owns" on both.
: "${CI_RELEASE_CONTRACT:=0}"

ci_print_versions()
{
    cmake --version
    command -v ninja >/dev/null 2>&1 && ninja --version
    c++ --version
    "${CLANG_FORMAT}" --version
    command -v clang-tidy >/dev/null 2>&1 && clang-tidy --version
    command -v glslc >/dev/null 2>&1 && glslc --version
    command -v glslangValidator >/dev/null 2>&1 && glslangValidator --version
    return 0
}

ci_format()
{
    find include src tests -type f \( \
        -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cxx' \
        \) -print0 | xargs -0 "${CLANG_FORMAT}" --dry-run -Werror
}

ci_configure()
{
    cmake --preset vcpkg -DFIRE_ENGINE_WARNINGS_AS_ERRORS="${WARNINGS_AS_ERRORS}"
}

ci_build()
{
    cmake --build build
}

ci_tidy()
{
    cmake --build build --target run-clang-tidy
}

ci_test()
{
    cmake --build build --target tests-full
}

# The RELEASE CONTRACT: the suite's `#ifdef NDEBUG` bodies, in the only configuration where they are
# real code.
#
# Every other stage here builds `Dev`, so those bodies compile to nothing — the cases run, pass, and
# assert precisely nothing. This stage exists because that is indistinguishable from a clean run.
#
# A GENUINE Release build, not a flag bolted onto the Dev tree: the behaviour under test belongs to
# the LIBRARY (a writer that asserts in Dev must return false under NDEBUG), so the library has to
# be the one compiled with NDEBUG. `vcpkg-release` puts it in its own binaryDir while sharing
# build/vcpkg_installed — see the preset's own description.
#
# Only `test_fire_engine` is built. It already pulls in `fireengine` and the shaders; the
# application adds no contract coverage. And only the tagged cases run: the whole Release suite
# would drag in the optimisation-sensitive physics goldens, which are a different question with
# different failure modes.
ci_release_contract()
{
    cmake --preset vcpkg-release -DFIRE_ENGINE_WARNINGS_AS_ERRORS="${WARNINGS_AS_ERRORS}"
    cmake --build --preset release-contract
    # Catch2 exits non-zero when a spec matches nothing, so a renamed tag fails here rather than
    # passing quietly; the hidden sentinel case inside the selection fails if this binary was not
    # built with NDEBUG. Between them, a green run means the contract was actually checked.
    ./build-release/test_fire_engine "[release-contract]"
}

# Run one named stage, composing prerequisites the same way CI does.
ci_run_stage()
{
    case "$1" in
        format) ci_format ;;
        configure) ci_configure ;;
        build) ci_configure && ci_build ;;
        tidy) ci_configure && ci_build && ci_tidy ;;
        test) ci_configure && ci_build && ci_test ;;
        release-contract) ci_release_contract ;;
        all)
            ci_format && ci_configure && ci_build && ci_tidy && ci_test || return $?
            # Appended rather than folded into the chain above: `all` means "every gate this
            # platform owns", and the Release contract is one Linux job by design.
            if [ "${CI_RELEASE_CONTRACT}" = "1" ]; then
                ci_release_contract
            fi
            ;;
        *)
            echo "unknown CI stage: $1" >&2
            return 2
            ;;
    esac
}

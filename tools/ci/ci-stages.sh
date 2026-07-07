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

: "${CLANG_FORMAT:=clang-format}"
: "${WARNINGS_AS_ERRORS:=ON}"

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

# Run one named stage, composing prerequisites the same way CI does.
ci_run_stage()
{
    case "$1" in
        format) ci_format ;;
        configure) ci_configure ;;
        build) ci_configure && ci_build ;;
        tidy) ci_configure && ci_build && ci_tidy ;;
        test) ci_configure && ci_build && ci_test ;;
        all) ci_format && ci_configure && ci_build && ci_tidy && ci_test ;;
        *)
            echo "unknown CI stage: $1" >&2
            return 2
            ;;
    esac
}

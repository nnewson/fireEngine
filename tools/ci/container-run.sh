#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

sync_source() {
  rsync -a --delete \
    --exclude /.git \
    --exclude /build \
    --exclude /vcpkg \
    --exclude /vcpkg_installed \
    /repo/ /work/fireEngine/
}

ensure_vcpkg() {
  if [ ! -d "${VCPKG_ROOT}/.git" ]; then
    rm -rf "${VCPKG_ROOT:?}"/*
    git clone https://github.com/microsoft/vcpkg "${VCPKG_ROOT}"
  else
    git -C "${VCPKG_ROOT}" fetch --depth=1 origin master
    git -C "${VCPKG_ROOT}" checkout --detach FETCH_HEAD
  fi
  "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
}

print_versions() {
  cmake --version
  ninja --version
  c++ --version
  clang-tidy --version
  clang-format-22 --version
  if command -v glslc >/dev/null 2>&1; then
    glslc --version
  fi
  glslangValidator --version
}

run_format() {
  find include src tests -type f \( \
    -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cxx' \
  \) -print0 | xargs -0 clang-format-22 --dry-run -Werror
}

run_configure() {
  cmake --preset vcpkg -DFIRE_ENGINE_WARNINGS_AS_ERRORS=ON
}

run_build() {
  cmake --build build
}

run_tidy() {
  cmake --build build --target run-clang-tidy
}

run_test() {
  cmake --build build --target tests-full
}

sync_source
print_versions

if [ "${mode}" = "shell" ]; then
  exec bash
fi

if [ "${mode}" != "format" ]; then
  ensure_vcpkg
fi

case "${mode}" in
  format)
    run_format
    ;;
  configure)
    run_configure
    ;;
  build)
    run_configure
    run_build
    ;;
  tidy)
    run_configure
    run_build
    run_tidy
    ;;
  test)
    run_configure
    run_build
    run_test
    ;;
  all)
    run_format
    run_configure
    run_build
    run_tidy
    run_test
    ;;
  *)
    echo "unknown local CI mode: ${mode}" >&2
    exit 2
    ;;
esac

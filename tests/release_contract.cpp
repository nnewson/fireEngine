#include <catch2/catch_test_macros.hpp>

// THE RELEASE CONTRACT, AND THE SENTINEL THAT PROVES IT WAS ACTUALLY CHECKED.
//
// A number of this suite's cases assert RELEASE behaviour: what a writer that asserts in a Dev
// build returns once `NDEBUG` compiles that assertion away. `ShadowRenderViewSet`'s writers are the
// clearest family — Dev stops at the assertion inside the writer, while under `NDEBUG` the writer
// must return `false` and clear the slot it addressed, which is the half that ships. Those bodies
// sit behind `#ifdef NDEBUG` and are tagged `[release-contract]`.
//
// Every preset in this repository builds `Dev`. Locally, in the Docker replica, and in every
// GitHub job before this one, those bodies therefore compiled to NOTHING — the cases ran, passed,
// and asserted precisely nothing, for as long as they have existed. That is what the Linux
// `release-contract` job exists to fix.
//
// This file is the sentinel, and it is here because a job that checks nothing looks exactly like a
// job that checks everything and finds no fault. Two failure modes are covered, by two different
// mechanisms:
//
//  1. THE SELECTION MATCHED NOTHING — a renamed or mistyped tag. Catch2 already exits non-zero when
//     a test spec matches no tests, so this needs no help from us.
//  2. THE BUILD WAS NOT A RELEASE BUILD. That one is silent: the filter matches the tagged cases,
//     they run, their guarded halves are empty, and the job reports success while proving nothing.
//     The case below fails outright in that situation, so the job cannot pass by accident.
//
// HIDDEN (`[.]`) on purpose. The Dev suites — `ctest` (`~[slow]`) and `tests-full` (no filter) —
// are both default runs, and Catch2 excludes hidden cases from those, so this does not fail the
// builds it is designed to detect. Explicitly selecting `[release-contract]`, which is the only
// thing the release job does, runs it. Do not remove the `[.]`, and do not give the tag to a case
// that has no `#ifdef NDEBUG` in it: the tag's meaning is "this case asserts something a Dev build
// cannot see".
TEST_CASE("the release contract is only checked where NDEBUG is defined", "[.][release-contract]")
{
#ifdef NDEBUG
    SUCCEED("NDEBUG is defined — the guarded bodies in this selection are real code");
#else
    FAIL("[release-contract] was selected from a build WITHOUT NDEBUG: every guarded body in this "
         "selection compiled to nothing, so the run proves nothing. Configure the vcpkg-release "
         "preset (or any Release build) and re-run.");
#endif
}

# Guard: a test case whose body is conditional on NDEBUG carries the `[release-contract]` tag, and
# nothing else does.
#
# The tag is what the Linux `release-contract` job selects, and it is the ONLY thing that selects
# it. An untagged `#ifdef NDEBUG` case is therefore not a weaker check, it is no check at all: every
# preset here builds `Dev`, so its guarded body compiles to nothing in every suite that runs it, and
# the case passes everywhere while asserting nothing anywhere. That is precisely the failure the tag
# was introduced to end, and it comes back silently the first time someone adds a conditional case
# without knowing the convention — which is the normal case, since the convention is invisible at
# the point where it matters.
#
# The other direction is checked too, and is not merely tidiness. `[release-contract]` means "this
# case asserts something a Dev build cannot see". A case carrying the tag without a conditional body
# runs identically in both configurations, so it pads the job's selection with work every other
# suite already covers and makes a green Release run look broader than it is.
#
# COMMENTS ARE STRIPPED FIRST, both forms. This file's subject is `#ifdef NDEBUG`, and so is the
# prose of nearly every case it guards: an unstripped scan is satisfied by a case that merely talks
# about release behaviour and — worse — by one whose real conditional has been commented out.
#
# Invoked as a CTest case; needs TESTS_DIR.

if(NOT DEFINED TESTS_DIR)
  message(FATAL_ERROR "TESTS_DIR must be set (path to tests/)")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/strip_glsl_comments.cmake")

set(offenders "")
set(tag "[release-contract]")
set(sentinel_file "release_contract.cpp")
set(sentinel_seen FALSE)
set(tagged_count 0)

file(GLOB_RECURSE test_sources "${TESTS_DIR}/*.cpp")
if(NOT test_sources)
  message(FATAL_ERROR
          "release contract guard: no .cpp files under ${TESTS_DIR} — the sweep found nothing to check, which is not a pass")
endif()

foreach(source IN LISTS test_sources)
  file(RELATIVE_PATH relative "${TESTS_DIR}" "${source}")
  file(READ "${source}" raw)
  strip_glsl_comments("${raw}" code)

  # Split the file into cases. CMake regexes cannot report a match POSITION, so the macro name
  # becomes the list separator and the text is split on it: element 1 is whatever precedes the first
  # case (includes, helpers) and is dropped; every later element is one case — its argument list
  # followed by its body, up to the start of the next case.
  #
  # Semicolons and backslashes go first, and both matter: `;` is CMake's list separator, so a case
  # containing one would be scanned in fragments; a trailing `\` escapes the separator itself.
  # Neither character can join two identifiers, so replacing them with a space cannot manufacture or
  # destroy a match below.
  string(REPLACE "\\" " " scan "${code}")
  string(REPLACE ";" " " scan "${scan}")
  string(REPLACE "TEST_CASE(" ";" scan "${scan}")

  set(index 0)
  foreach(case IN LISTS scan)
    math(EXPR index "${index} + 1")
    if(index EQUAL 1)
      continue()
    endif()

    # Argument list and body are separated at the first line that STARTS with `{`, not at the first
    # `)`: 36 case names in this suite contain a parenthesis and one contains a brace, so a
    # punctuation-based split would read part of a name as body (and then miss its tag). Allman
    # braces make the line-initial `{` reliable, and they are themselves CI-gated by clang-format.
    string(FIND "${case}" "\n{" brace)
    if(brace LESS 0)
      list(APPEND offenders
           "${relative}: a TEST_CASE( has no body brace on its own line — the guard cannot tell its tags from its body")
      continue()
    endif()
    string(SUBSTRING "${case}" 0 ${brace} arguments)
    string(LENGTH "${case}" case_length)
    math(EXPR body_start "${brace} + 2")
    math(EXPR body_length "${case_length} - ${body_start}")
    string(SUBSTRING "${case}" ${body_start} ${body_length} body)

    # The case's NAME, for the report — the first string literal, by Catch2's signature.
    set(name "<unnamed>")
    if(arguments MATCHES "\"([^\"]*)\"")
      set(name "${CMAKE_MATCH_1}")
    endif()

    string(FIND "${arguments}" "${tag}" tag_position)
    if(tag_position GREATER_EQUAL 0)
      set(tagged TRUE)
      math(EXPR tagged_count "${tagged_count} + 1")
    else()
      set(tagged FALSE)
    endif()

    # A preprocessor conditional ON NDEBUG, in any spelling. `#if !defined(NDEBUG)` counts: what
    # matters is that the case's behaviour differs between the two configurations, not which half is
    # which.
    if(body MATCHES "#[ \t]*(ifdef|ifndef|if)[^\n]*NDEBUG")
      set(conditional TRUE)
    else()
      set(conditional FALSE)
    endif()

    if(conditional AND NOT tagged)
      list(APPEND offenders
           "${relative}: \"${name}\" has an NDEBUG-conditional body but is not tagged ${tag} — nothing builds it as real code, so it asserts nothing in any configuration")
    elseif(tagged AND NOT conditional)
      list(APPEND offenders
           "${relative}: \"${name}\" is tagged ${tag} but has no NDEBUG conditional — the tag means 'this asserts something a Dev build cannot see', and every other suite already covers this case")
    endif()

    # The sentinel: hidden, tagged, conditional. Its `[.]` is what keeps it out of the Dev suites,
    # and removing the `[.]` fails them loudly — but removing the CASE is silent, and leaves the job
    # unable to tell a Release build from a Dev one again.
    if(relative STREQUAL "${sentinel_file}" AND tagged AND conditional AND arguments MATCHES "\\[\\.\\]")
      set(sentinel_seen TRUE)
    endif()
  endforeach()
endforeach()

if(NOT sentinel_seen)
  list(APPEND offenders
       "tests/${sentinel_file} has no hidden ([.]) ${tag} case with an NDEBUG conditional — that case is what fails the job when the binary was not built with NDEBUG, and without it a Dev build passes the selection silently")
endif()

# A selection of nothing is not a green run. Catch2 catches that at job time (it exits non-zero when
# a spec matches no tests), but this fails in the build that removed the last tag, where the cause is
# still in front of whoever caused it.
if(tagged_count EQUAL 0)
  list(APPEND offenders "no test case carries ${tag} — the Linux release-contract job now selects nothing")
endif()

if(offenders)
  string(REPLACE ";" "\n  " report "${offenders}")
  message(FATAL_ERROR "release contract guard failed:\n  ${report}")
endif()

message(STATUS
        "release contract guard: ${tagged_count} case(s) tagged ${tag}, each with an NDEBUG-conditional body, plus the hidden sentinel")

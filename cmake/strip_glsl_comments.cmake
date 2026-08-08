# `strip_glsl_comments(in_text out_var)` — remove both comment forms before a textual guard reads a
# file. Shared by the shader guards rather than copied into each, because the block-comment pattern
# below is easy to get subtly wrong and a wrong version fails open.
#
# Every shader guard asks whether the CODE does something, and an unstripped match is satisfied by
# commented-out text — which is exactly how a call gets disabled while appearing to survive. Line
# comments alone are not enough: wrapping a whole path in /* ... */ keeps its calls visible to a
# naive match, and can hide a shared file's own declarations so every later check passes vacuously.
# Both gaps were found by mutation-testing a guard, which is the only way to know a textual check
# bites.
#
# The comment syntax is identical in GLSL and C++, so this serves the C++-side checks too.

# Block first, so a `//` inside a block cannot survive it. The block pattern is the classic
# non-greedy-free form — CMake's regex has no lazy quantifier, and `/\*.*\*/` would swallow
# everything between the FIRST and LAST comment in the file, silently blanking the input and making
# every check "fail" for the wrong reason. The line pattern anchors on start-of-string as well as
# newline, so a comment at byte zero is stripped too.
function(strip_glsl_comments in_text out_var)
  string(REGEX REPLACE "/\\*[^*]*\\*+([^/*][^*]*\\*+)*/" "" stripped "${in_text}")
  string(REGEX REPLACE "//[^\n]*" "" stripped "${stripped}")
  set(${out_var} "${stripped}" PARENT_SCOPE)
endfunction()

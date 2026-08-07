// THE PCF kernel's taps — one definition, consumed by both the shader that samples them and the test
// that checks their support.
//
// Not a header in either language: it is a list of `POISSON_TAP(x, y)` invocations, and each side
// defines that macro to build whatever it needs (a `vec2` array initialiser in GLSL, a
// `std::array<float, 2>` in the test). That is the only arrangement in which the test pins what
// SHIPS. A copied table in the test proves the copy is normalised while a production tap can be
// edited back outside unit support with everything still green — which is exactly how a guard becomes
// decoration.
//
// NORMALISED TO UNIT SUPPORT (SH-07): the classic table these come from has a maximum length of
// 1.2339, with 8 of its 16 taps outside the unit disc, so a kernel asked for "radius R" sampled out
// to 1.234R. The bias law clears `filterRadiusTexels + 1`, so the sampler and the law would have
// meant different things by the same number. Dividing through by the support fixes that and leaves
// the distribution untouched. `ShadowBias.ThePcfKernelHasUnitSupport` fails if any edit here breaks
// it.
//
// Commas separate the invocations and there is deliberately NO trailing comma: GLSL rejects one in an
// array initialiser.
POISSON_TAP(-0.76342630, -0.32340690),
POISSON_TAP( 0.76631937, -0.62313577),
POISSON_TAP(-0.07632843, -0.75319272),
POISSON_TAP( 0.27956106,  0.23816350),
POISSON_TAP(-0.74224974,  0.37093962),
POISSON_TAP(-0.66084860, -0.71245785),
POISSON_TAP(-0.31020785,  0.22429795),
POISSON_TAP( 0.79003047,  0.61306758),
POISSON_TAP( 0.35920391, -0.79025054),
POISSON_TAP( 0.43554244, -0.38392241),
POISSON_TAP(-0.21473556, -0.33950832),
POISSON_TAP( 0.64183039,  0.15471019),
POISSON_TAP(-0.19603055,  0.80803882),
POISSON_TAP(-0.65976039,  0.74102609),
POISSON_TAP( 0.16195482,  0.63732328),
POISSON_TAP( 0.11656363, -0.11427525)

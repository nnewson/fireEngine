# Code Review

## Tier 0 — Math & value types

Reviewed: 18 July 2026

Scope: `include/fire_engine/math/`, with relevant tests and production call sites checked to assess
the impact of the APIs. The review focuses on correctness, refactoring, simplification,
optimisation, standardisation, and appropriate use of C++23.

### Overall recommendation

Do not replace the entire library wholesale. The vector and matrix storage is fundamentally sound,
but a targeted foundational redesign is justified around comparison, inversion, normalisation,
quaternion invariants, and transform semantics.

The recommended direction is:

1. Fix the comparison, normalisation, and matrix-inversion correctness issues.
2. Introduce an invariant-preserving rotation type.
3. Separate affine transforms from general projective matrices.
4. Standardise the remaining small-vector and matrix APIs.

Retaining a small in-house math library is preferable to replacing it blindly with GLM or Eigen.
The engine benefits from explicitly owning its physics, transform, clip-space, and GPU ABI
conventions. The main problems are semantic contracts rather than missing arithmetic machinery.

### Findings

#### 1. High: `Mat3::inverse()` rejects valid small transforms

[`mat3.hpp`](include/fire_engine/math/mat3.hpp) uses an absolute determinant threshold:

```cpp
if (det <= eps && det >= -eps)
{
    return Mat3{};
}
```

A uniform scale of `1e-5` is perfectly conditioned but has determinant `1e-15`, so its inverse is
incorrectly returned as the zero matrix.

This creates a concrete inconsistency in [`vdpm.cpp`](src/graphics/vdpm.cpp):

- `makeVdpmViewParams()` deliberately uses the scale-invariant test
  `abs(det) > 1e-6 * sigmaMax^3` to decide whether the transform is invertible.
- A tiny uniform transform passes that test.
- `linear.inverse()` then applies its absolute determinant threshold and returns zero.
- `cameraObj` is consequently wrong even though `coneUsable` is true.

Recommended change:

- Add `tryInverse(relativeTolerance) -> std::optional<Mat3>`.
- Scale the matrix by its maximum absolute component before calculating the determinant.
- Use `double` intermediates.
- Stop using the zero matrix as an ambiguous failure sentinel.
- Make physics callers explicitly assert or propagate expected invertibility.

This should be the first change made.

#### 2. High: `approxEqual()` accepts NaNs as equal

The same comparison pattern appears in `VecBase`, `Mat3`, `Mat4`, and `Quaternion`:

```cpp
const float diff = a - b;
if (diff > eps || diff < -eps)
{
    return false;
}
```

Both comparisons are false when `diff` is NaN, so NaN compares approximately equal to anything.
Equal-sign infinities also subtract to NaN and pass accidentally. This can hide exactly the
numerical failures that tests should expose.

Create one scalar comparison authority, for example:

```cpp
almostEqual(float a, float b, float absoluteTolerance, float relativeTolerance)
```

It should:

- Return true immediately for `a == b`, including equal infinities.
- Return false for any remaining non-finite inputs.
- Use a combined relative and absolute tolerance for finite inputs.

All vector, matrix, and quaternion approximate comparisons should delegate to it.

#### 3. High: rotation quaternions do not enforce their required invariant

`Quaternion` is freely constructible and mutable, while much of its API assumes unit length:

- `rotate()`
- `fromVectors()`
- `fromAxisAngle()`
- `slerp()`
- `toMat4()`

For example, `fromAxisAngle({0, 0, 0}, angle)` produces a non-unit value that is not a valid
rotation. A non-unit axis silently introduces invalid rotation behaviour.

The principled redesign is to separate:

- `Quaternion`: unrestricted four-component algebra, if it is genuinely needed.
- `UnitQuaternion` or `Rotation3`: a private representation with factories that preserve unit
  length.

Transforms, animation, physics orientation, and rendering should use the rotation type. It should
also provide rotation-aware comparison because `q` and `-q` represent the same rotation.

A smaller intermediate change would normalise factory inputs and give fast preconditioned forms
explicit names such as `fromUnitAxisAngle()` and `fromUnitVectors()`.

#### 4. Medium: norm calculation is not numerically robust

`VecBase::magnitude()` and `Quaternion::magnitude()` square components directly. Large finite
components can overflow during squaring, while very small components can underflow.

Use a scaled norm implementation:

1. Find the largest absolute component.
2. Divide all components by it.
3. Calculate the norm in that safe range.
4. Rescale the result.

Define and test explicit behaviour for zero and non-finite inputs.

The current `float_epsilon` should also be split into values with distinct meanings:

- approximate-comparison absolute tolerance;
- approximate-comparison relative tolerance;
- minimum normalisable length;
- operation-specific geometric degeneracy thresholds.

One global epsilon is not meaningful across all of those operations.

#### 5. Medium: "bitwise equality" is not bitwise

The equality comments and `bitwiseEqual()` methods use ordinary floating-point `==`. Consequently:

- `-0.0f == +0.0f`;
- NaNs never equal themselves.

This is exact component-wise IEEE equality, not bitwise equality.

Delete `bitwiseEqual()` if exact component equality is all that is required; it currently duplicates
`operator==`. If determinism diagnostics need real bit comparison, implement it explicitly using
component-wise `std::bit_cast<std::uint32_t>`.

#### 6. Medium: affine and projective operations are mixed together

`Mat4::transformPoint()` explicitly warns that it drops homogeneous `w` rather than performing a
perspective divide. The same class also owns projection-matrix factories, making affine and
projective operations easy to confuse.

Introduce an `Affine3` type containing a `Mat3` linear part and a translation. It should provide:

- `transformPoint()`;
- `transformVector()`;
- `transformNormal()`;
- affine inversion;
- direct TRS construction and composition.

Keep `Mat4` for general/projective matrices and provide a distinct `projectPoint()` operation that
performs the perspective divide and explicitly handles a near-zero `w`.

This would also replace repeated expressions such as:

```cpp
Mat4::translate(position) * rotation.toMat4() * Mat4::scale(scale)
```

with one direct affine/TRS construction.

#### 7. Medium: projection conventions are hidden in generic names

`Mat4::perspective()` and `Mat4::ortho()` bake in Vulkan's flipped Y axis, right-handed view, and
`[0, 1]` depth range. The implementations are reasonable, but their generic names hide important
conventions.

Make the convention visible, for example:

```cpp
perspectiveVulkanRH_ZO(...)
orthographicVulkanRH_ZO(...)
```

Alternatively, move clip-space construction into the rendering or camera layer, possibly behind an
explicit clip-space convention policy.

#### 8. Medium: conversion authority is duplicated and headers are tightly coupled

`Mat3::fromQuaternion()` rotates three basis vectors, while `Quaternion::toMat4()` independently
contains a direct conversion formula. This duplicates conversion authority, can drift, and makes
`Mat3` include `Quaternion`, which in turn includes `Mat4`.

Prefer one conversion path:

```text
Rotation3::toMat3()
Mat4::fromAffine(linear, translation)
```

Move cross-type conversions to a dedicated transform/rotation header or implementation file. A
direct quaternion-to-`Mat3` formula is also cheaper than rotating three basis vectors.

### Smaller standardisation improvements

- Add unary vector negation and scalar-left multiplication: `-v` and `2.0f * v`.
- Add `operator[]`, `data()`, `size()`, and possibly `std::span` access to vectors.
- Give `Vec2` `x()`/`y()` accessors as well as `s()`/`t()` because it is used outside texture
  coordinates.
- Replace raw arrays with `std::array<float, N>`.
- Use `std::size_t` for indices and optionally assert bounds in debug builds.
- Add size, standard-layout, and trivial-copyability assertions for packed math types.
- Standardise matrix compound operators according to the project's rule that compound assignment is
  the primitive operation.
- Replace the hand-written pi constant with `std::numbers::pi_v<float>`.
- Move `kCameraMaxPitch` out of the general math constants header.
- Replace the `0.99f` view-basis threshold by choosing the canonical axis least aligned with the
  forward vector. This is deterministic and maximises the cross-product magnitude.
- Replace the always-`w = 1` `Vec3` to `Vec4` lift with named point/direction factories, or use
  `Affine3::transformPoint()` and `transformVector()` so the distinction is encoded in the API.
- Preserve the C++23 two-argument matrix subscript; it is a good use of C++23.

### Performance assessment

Do not introduce explicit SIMD without benchmarks. The fixed 3x3 and 4x4 loops should already
unroll under `-O2`, while forced SIMD alignment could increase storage and complicate GPU and physics
layouts.

The worthwhile obvious optimisations are:

- direct quaternion-to-matrix construction;
- direct TRS/affine construction;
- avoiding repeated matrix products for transforms;
- robust scaled normalisation;
- consistently passing small trivially-copyable vector values according to a documented convention.

### Testing improvements

The current vector tests are repetitive, while some of the more important numerical properties are
under-tested. Consolidate common vector behaviour with templated Catch2 helpers and add targeted
property/invariant tests for:

- NaN and infinity comparison behaviour;
- normalisation across logarithmically distributed magnitudes;
- inversion of uniformly tiny and large matrices;
- random well-conditioned `Mat3` inverse round trips;
- singular and near-singular inverse failure;
- `q` and `-q` rotation equivalence;
- quaternion factory unit-length invariants;
- `slerp()` endpoint, shortest-path, and unit-length properties;
- matrix/quaternion conversion round trips;
- explicit Vulkan projection near/far and Y-axis mapping;
- standard-layout, trivial-copyability, and exact size of packed types.

The existing `Mat3` tests particularly need expansion: inversion is a critical operation, but the
current suite does not cover scale invariance or the absolute-determinant failure above.

### Recommended implementation sequence

#### Phase 1: correctness foundation

- Add regression tests for NaN comparison and tiny, well-conditioned matrix inversion.
- Add the shared scalar comparison utility.
- Implement robust scaled norms.
- Add scale-aware `Mat3::tryInverse()` and migrate its callers.
- Correct or remove the misleading bitwise-equality API.

#### Phase 2: rotation redesign

- Introduce `UnitQuaternion` or `Rotation3`.
- Centralise quaternion-to-matrix conversion.
- Migrate transform, animation, rendering, and physics orientation users.
- Add rotation-aware comparison and factory-invariant tests.

#### Phase 3: transform and API redesign

- Introduce `Affine3` and direct TRS construction.
- Separate affine point/vector/normal transformation from projective transformation.
- Make projection conventions explicit.
- Standardise the vector and matrix access/operator surface.
- Consolidate repetitive math tests into reusable typed/property tests.

## Tier 1 — Handles, limits, tunables

Reviewed: 19 July 2026

Scope:

- [`graphics/gpu_handle.hpp`](include/fire_engine/graphics/gpu_handle.hpp)
- [`graphics/gpu_limits.hpp`](include/fire_engine/graphics/gpu_limits.hpp)
- [`render/constants.hpp`](include/fire_engine/render/constants.hpp)
- [`physics/physics_handle.hpp`](include/fire_engine/physics/physics_handle.hpp)
- [`collision/collider_id.hpp`](include/fire_engine/collision/collider_id.hpp)
- [`core/log.hpp`](include/fire_engine/core/log.hpp)

The backing allocators, resource lookups, broadphase implementations, shaders, tests, and
representative production call sites were also checked where they define the real contract of these
vocabulary types.

### Overall recommendation

Do not merely standardise spelling in this tier. Two small foundational rewrites are justified:

1. Replace the mixture of enum handles, raw table indices, generational handles, and the
   graphics-owned slot pool with one core identity facility that has explicit checked semantics.
2. Replace the logger's open string category plus per-call rule search with a closed category enum
   and an indexed immutable configuration.

The limits and tunables do not need a wholesale rewrite, but GPU-visible limits need a generated
cross-language authority rather than matching comments, and derived render constants should be
calculated rather than repeated.

### Findings

#### 1. High: texture generations are not enforced by resource lookup or release

[`gpu_handle.hpp`](include/fire_engine/graphics/gpu_handle.hpp) describes generations as making a
stale handle "detectably invalid on lookup". The backing texture pool does track generations, but
most [`Resources`](src/render/resources.cpp) operations discard them:

```cpp
return *textures_[handleIndex(handle)].view;
```

The same applies to image, sampler, format, mip-level, face-view, bindless-registration, and release
paths. `validTexture()` exists, but the owner does not enforce it. In particular,
`releaseTexture(staleHandle)` can destroy a newer texture that reused the same index and then release
its slot again. A stale lookup silently aliases the replacement instead of being detected.

This defeats the main correctness claim of the generational handle design.

Recommended change:

- Centralise texture resolution in `Resources::requireTexture()` / `resolveTexture()`.
- Validate both index bounds and generation before every dereference or release.
- Make the throwing/nullable contract explicit rather than keeping unchecked accessors `noexcept`.
- Make `releaseTexture()` reject a stale or already-released handle.
- Add stale-after-reuse and double-release tests against `Resources`, not only against the isolated
  slot pool.

VDPM front/mesh lookup already validates generation before dereferencing its table; use that as the
minimum owner-side contract.

#### 2. High: GPU layout limits have no machine-enforced C++/GLSL authority

[`gpu_limits.hpp`](include/fire_engine/graphics/gpu_limits.hpp) calls itself the shared source of
truth, but shader-visible values are independently repeated as literals:

- `kMaxJoints = 64` versus `mat4 joints[64]` in two vertex shaders;
- `kMaxMorphTargets = 8` versus `vec4 weights[2]`;
- `kMaxLights = 8` versus `MAX_LIGHTS = 8`;
- shadow cascade/spot/self-shadow counts and the total matrix count;
- `kMaxParticleEmitters = 4` versus `emitters[4]` in two particle shaders;
- `kSsaoKernelSize = 16` versus `KERNEL_SIZE = 16`.

The C++ layout assertions prove only the CPU layout. A one-sided constant change can still compile
both languages successfully while making array bounds, UBO sizes, or shader loops disagree.

The principled fix is to generate both a C++ header and a GLSL include from one small limits
manifest. Every affected shader should include the generated GLSL definitions. A build-time parser
test comparing literals is weaker: it detects drift after duplicating the values, rather than
making drift impossible.

Keep device-dependent capacity validation in `Device`, but generate the compile-time ABI values.
This is a correctness boundary, not a documentation convention.

#### 3. High: `ColliderId` registration state is inconsistent between broadphases

The value type in [`collider_id.hpp`](include/fire_engine/collision/collider_id.hpp) is reasonable,
but the ownership protocol around it has diverged:

- `SweepAndPruneBroadPhase::removeCollider(ColliderId)` clears the removed collider's ID.
- `DynamicAabbTreeBroadPhase::removeCollider(ColliderId)` destroys the proxy but leaves the
  collider carrying a valid-looking ID.
- `SweepAndPruneBroadPhase::clear()` clears every collider's ID.
- `DynamicAabbTreeBroadPhase::clear()` clears only its own containers, leaving all former colliders
  carrying valid-looking IDs.
- SAP treats adding an already-registered collider as an idempotent operation; the dynamic tree can
  register another proxy and replace the collider's ID while the old proxy remains.

The stale ID is not cosmetic: `Collider` move operations use the ID as the registered/unregistered
precondition. After the dynamic tree's ID-based removal or `clear()`, an unregistered collider can
remain unnecessarily immovable, and a double registration can leave duplicate broadphase state.

Immediate fix:

- Make both implementations clear the payload's ID on every removal and clear operation.
- Define and test one repeated-add policy.
- Add parity tests that run the same registration lifecycle against both broadphases.

Larger simplification: make the ID an internal broadphase proxy identity, rename it
`BroadPhaseProxyId`, and remove the public ID-based removal overload if production only removes by
`Collider&`. The public abstraction should not expose two removal routes whose state-maintenance
requirements can drift.

#### 4. High: handle packing silently aliases invalid inputs

Both `makeHandle()` and `PhysicsHandle::make()` mask their inputs:

```cpp
((generation & kHandleGenerationMask) << kHandleIndexBits) |
    (index & kHandleIndexMask)
```

Out-of-range values therefore alias an unrelated valid handle instead of being rejected. There is
also a reserved-value collision:

```cpp
makeHandle<TextureHandle>(kHandleIndexMask, kHandleGenerationMask) == NullTexture
```

The current test describes that maximum packed pair as valid even though it equals the null
sentinel. Separately, the pool deliberately wraps an eight-bit generation after 255 releases, at
which point an old surviving handle becomes valid again. That contradicts the unqualified stale
handle guarantee in the API comments.

Recommended options, in preference order:

1. Use a 64-bit CPU-side generational handle. GPU bindless indices can still be extracted and sent
   separately as 32-bit values.
2. If 32-bit storage is important, give each handle family capacity-specific index/generation bit
   counts and permanently retire a slot before its generation wraps.
3. At minimum, check index/generation bounds, reserve the null bit pattern explicitly, and return an
   error or fail a programmer precondition rather than truncating.

`GenerationalSlotPool` should also track occupancy, reject an out-of-range or double release, and
include occupancy in `valid()`. Its current generation-only validity test can report a free slot as
valid if the caller knows the next generation.

#### 5. Medium: `gpu_handle.hpp` combines incompatible handle models

`BufferHandle`, `PipelineHandle`, and `DescriptorSetHandle` are raw monotonically-growing table
indices. `TextureHandle`, `VdpmMeshHandle`, and `VdpmFrontHandle` are generational. All nevertheless
share unconstrained `handleIndex()`, `handleGeneration()`, and `makeHandle()` templates.

This already creates a latent buffer/pipeline lookup failure: those raw handles are stored as the
full table index, but `Resources` later passes them through `handleIndex()`, which masks off every
bit above bit 23. Reaching 2^24 entries would alias slot zero rather than fail at the capacity
boundary.

The null/default semantics are inconsistent too:

- a default-initialised GPU enum handle has value zero, which is valid;
- a default-initialised physics handle has value zero, which is null;
- GPU null is `UINT32_MAX`;
- generational generations start at one, but the public APIs can manufacture generation zero.

This makes generic initialisation hazardous. Several per-frame arrays explicitly list two null
handles; if their extent grows, remaining elements value-initialise to the valid handle zero.

A small zero-overhead strong-handle rewrite is justified. Use separate core templates or policies
for raw index handles and generational handles, with:

- default construction producing the invalid value consistently;
- `explicit operator bool()` or `hasValue()` for the non-null test;
- owner-only or named `fromRaw()` construction;
- constrained index/generation operations available only on the correct handle family;
- standard-layout, trivial-copyability, size, and hashing assertions.

This would also remove the repeated `NullBuffer`, `NullTexture`, etc. variables in favour of a
uniform default state without losing type safety.

#### 6. Medium: `kMaxFramesInFlight` is presented as a limit but behaves as the literal value two

Changing [`kMaxFramesInFlight`](include/fire_engine/graphics/gpu_limits.hpp) currently breaks
unrelated assumptions:

- TAA selects the previous slot with `(kMaxFramesInFlight - 1) - current`, which works only for
  exactly two slots.
- descriptor history uses the same reflection formula;
- numerous `std::array<Handle, kMaxFramesInFlight>` members are initialised with exactly two null
  handles, so a third slot becomes handle zero rather than null;
- some code mixes signed `int` loops with unsigned Vulkan counts and `std::size_t` extents.

Either declare and enforce two as an architectural invariant with `static_assert`, or make the ring
logic genuinely general:

```cpp
const auto previous = (current + frameCount - 1) % frameCount;
```

Use a filled-array helper or a default-null strong handle so every slot is initialised correctly.
The general solution is preferable because the constant's name and documentation advertise a
capacity, not a hard-coded double-buffering protocol.

#### 7. Medium: disabled logging performs allocation and a linear string-rule search

Every call to `log::debug/info/...` first reaches `levelFor()`, which lowercases the category into a
new `std::string` and scans the rules vector in reverse. Therefore even a disabled debug statement
can allocate. The built-in categories are closed, static, and already lowercase, so this work buys
nothing on the normal path.

`Category` is also an open aggregate containing a `string_view`; a category created from temporary
storage could dangle. There are no production custom categories to justify that openness.

Replace it with:

```cpp
enum class Category : std::uint8_t { App, General, Gltf, Physics, Ragdoll, Render, Count };
```

Parse `FE_LOG` once into `std::array<Level, categoryCount>`. Then `enabled()` is one bounds-checked
or asserted array lookup with no allocation or search. Keep a constexpr category-name table only
for parsing and output.

Move environment parsing, configuration storage, mutex ownership, and output into a `.cpp`. The
header needs the formatting templates, but it does not need to expose `Rule`, `Config`, trimming,
case conversion, environment access, `FILE*` output, or mutex implementation to every logging
caller.

#### 8. Medium: physics identity is coupled to graphics and duplicates packing logic

[`physics_handle.hpp`](include/fire_engine/physics/physics_handle.hpp) includes
`graphics/gpu_handle.hpp` solely to reuse bit constants, while `PhysicsWorld` uses a
`graphics/GenerationalSlotPool`. Identity allocation is not a graphics concept. The dependency also
allowed the packing implementation to be copied rather than shared, so fixes to one path can drift
from the other.

Move the checked slot pool and generational handle primitive into `core/` (or a neutral
`identity/` module), then define the GPU and physics tags on top.

Also rename `PhysicsHandle::valid()`. It means only "non-null"; a destroyed stale handle still
returns true until checked by `PhysicsWorld`. `hasValue()`, `isNull()`, or explicit `operator bool()`
states the local fact accurately, while `PhysicsWorld::contains(handle)` can express owner-validated
liveness.

The public raw-value constructor is currently used to decode packed event keys and to fabricate
test handles. Prefer a named `fromRaw()` boundary, or store/compare the handles directly through a
hash specialisation, so ordinary callers cannot accidentally manufacture a plausible live handle.

#### 9. Medium: derived render constants are repeated rather than derived

Several constants in [`gpu_limits.hpp`](include/fire_engine/graphics/gpu_limits.hpp) and
[`constants.hpp`](include/fire_engine/render/constants.hpp) encode relationships manually:

- `kShadowSpotMatrixBase` is the literal `4` rather than
  `kShadowCascadeMatrixBase + kShadowCascadeCount`;
- cubemap mip counts repeat values derivable from their extents;
- `kShadowTotalMatrixCount` depends on several signed and unsigned values with no range assertion;
- projection, shadow range, TAA sample count, and mip settings have no compile-time relational
  validation.

Use C++23/standard-library derivation where possible, for example `std::bit_width(extent)` for a
full power-of-two mip chain. Add `static_assert`s or a `consteval` validator for relationships such
as positive near planes, far > near, non-zero TAA cycle, power-of-two extents, and non-overlapping
shadow matrix ranges.

Group defaults by domain (`CameraDefaults`, `ShadowDefaults`, `IblDefaults`, etc.) or at least put
them in nested namespaces. The current global `k...` list is manageable today, but grouping would
make unit conventions and validation substantially clearer without imposing runtime cost.

#### 10. Low: logger exception and output contracts need tightening

`parseLevel()` is declared `noexcept` but calls helpers that allocate `std::string`; allocation
failure therefore terminates instead of propagating. Either remove `noexcept` or rewrite parsing as
allocation-free comparisons.

`write()` also performs several stdio calls while holding the mutex and calls `levelName()` twice.
Build the complete line first and issue one write while locked. This is not currently a meaningful
frame-time problem, but it gives the atomic-line contract a simpler implementation.

If expensive log arguments become common, consider a lazy form because ordinary function
arguments are evaluated before the logger can check `enabled()`. The existing explicit
`if (log::enabled(...))` guard remains appropriate for the few expensive diagnostics.

#### 11. Low: limit types should express their use consistently

The limit headers mix `int`, `uint32_t`, and `std::size_t`, causing repeated casts and signed loop
indices. Prefer:

- `std::size_t` for C++ array extents and container capacities;
- `std::uint32_t` for values copied into Vulkan/GLSL or used as GPU counts;
- an explicitly checked conversion at the boundary between them.

Do not change types mechanically where they are shader ABI fields; make the intended domain part of
the declaration and assert that narrowing is safe.

### Per-file assessment

- **`graphics/gpu_handle.hpp`: rewrite recommended.** The enum types are lightweight, but the file
  conflates raw and generational identities, cannot make default construction null, silently masks
  overflow, and permits a valid packed value to collide with null.
- **`graphics/gpu_limits.hpp`: structural refactor recommended.** Keep the conceptual split below
  `render/`, but generate its shader-visible subset for both languages and derive related values.
- **`render/constants.hpp`: targeted refactor.** Values themselves are reasonable; organise them by
  domain, derive mip/range relationships, and add compile-time validation.
- **`physics/physics_handle.hpp`: fold into the core handle redesign.** The tag-based type safety is
  good and should be preserved.
- **`collision/collider_id.hpp`: value representation can stay.** Its registration ownership and
  lifecycle need clarification/fixing; consider narrowing it to an internal broadphase proxy ID.
- **`core/log.hpp`: small rewrite recommended.** Preserve compile-time checked format strings and
  the simple public calls, but use enum categories, indexed configuration, and a `.cpp` core.

### Performance assessment

The handle redesign should remain zero-overhead: a strong wrapper around one integer, trivially
copyable, passed by value. Generation validation adds one table bounds check and integer comparison
at owner lookup; that is cheaper than debugging a stale-resource alias and is not in shader
execution.

The only likely runtime win in this tier is the logger category rewrite. It removes allocation,
lowercasing, and a vector scan from every disabled-log check. No custom allocator, lock-free logger,
or asynchronous logging queue is justified for the project's current diagnostic volume.

Compile-time impact should improve by moving most logger implementation and its heavy parsing/stdio
dependencies out of the header.

### Testing improvements

Add or strengthen tests for:

- stale texture lookup and release after a slot is reused;
- double release and out-of-range release in `GenerationalSlotPool`;
- generation exhaustion without stale-handle resurrection;
- checked rejection of oversized index/generation inputs;
- the packed maximum never colliding with the null representation;
- default construction being null for every public handle family;
- raw-index handle capacity failure before bit 24 is lost;
- identical add/remove-by-ID/remove-by-reference/clear/re-add behaviour across both broadphases;
- frame-ring previous-slot selection for ring sizes 1, 2, and 3;
- all-null initialisation of every frame-ring handle array;
- logger parsing of whitespace, invalid levels, wildcard/global rules, repeated rules, and last-rule
  precedence;
- allocation-free disabled logging, if an allocator-counting test facility is introduced;
- generated GLSL limits being consumed by every shader that declares a mirrored array.

The current generation-wrap test explicitly blesses stale aliasing after 255 cycles. Replace it
with the chosen no-alias contract rather than retaining that behaviour as an accepted limitation.

### Recommended implementation sequence

#### Phase 1: close current correctness holes

- Enforce texture generation at every resource lookup and release.
- Fix dynamic-tree collider ID clearing, repeated registration, and broadphase parity tests.
- Derive the shadow matrix ranges and add immediate compile-time relationship assertions.
- Add a build-enforced C++/GLSL limits authority.

#### Phase 2: identity foundation

- Introduce neutral raw-index and generational strong-handle primitives.
- Move `GenerationalSlotPool` out of `graphics/`, add occupancy tracking, and reject invalid release.
- Choose a no-resurrection generation policy and migrate GPU/physics handle families.
- Make default/null and owner-validated-liveness semantics consistent.

#### Phase 3: configuration and diagnostics cleanup

- Make frame-ring logic independent of the literal value two and replace partial array
  initialisers.
- Derive mip counts and validate render-default relationships at compile time.
- Replace string categories/rules with enum categories and an indexed logger config.
- Move logger parsing/output state to a `.cpp` and add parser/precedence tests.

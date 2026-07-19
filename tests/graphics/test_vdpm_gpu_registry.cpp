#include <catch2/catch_test_macros.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/vdpm_gpu_registry.hpp>
#include <fire_engine/math/mat4.hpp>

using namespace fire_engine;

// The Vulkan-free half of the GPU-driven VDPM registration seam (rendering-spine #3, Stage B5b):
// the work-request parameter comparison the renderer dedups on, and the identity-handle packing.
// Runs in normal CI (no GPU) — the manager's device-bound half is covered by the [.][gpu] tests.

namespace
{

VdpmWorkRequest baseRequest()
{
    VdpmWorkRequest r;
    r.front = makeHandle<VdpmFrontHandle>(3, 1);
    r.world = Mat4::identity();
    r.uvScale = 1.0f;
    r.normalScale = 0.5f;
    r.tangentScale = 0.25f;
    r.rasterBackfaceCulling = true;
    return r;
}

} // namespace

TEST_CASE("VdpmWorkRequest::sameParams matches identical requests", "[vdpm][gpu-registry]")
{
    const VdpmWorkRequest a = baseRequest();
    VdpmWorkRequest b = baseRequest();
    REQUIRE(a.sameParams(b));
    REQUIRE(b.sameParams(a)); // symmetric
}

TEST_CASE("VdpmWorkRequest::sameParams distinguishes every field", "[vdpm][gpu-registry]")
{
    const VdpmWorkRequest a = baseRequest();

    SECTION("front handle")
    {
        VdpmWorkRequest b = baseRequest();
        b.front = makeHandle<VdpmFrontHandle>(4, 1);
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("front generation (same index, recycled slot)")
    {
        VdpmWorkRequest b = baseRequest();
        b.front = makeHandle<VdpmFrontHandle>(3, 2);
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("world transform")
    {
        VdpmWorkRequest b = baseRequest();
        b.world[0, 3] = 5.0f;
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("uv scale")
    {
        VdpmWorkRequest b = baseRequest();
        b.uvScale = 2.0f;
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("normal scale")
    {
        VdpmWorkRequest b = baseRequest();
        b.normalScale = 0.6f;
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("tangent scale")
    {
        VdpmWorkRequest b = baseRequest();
        b.tangentScale = 0.3f;
        REQUIRE_FALSE(a.sameParams(b));
    }
    SECTION("raster cull policy")
    {
        VdpmWorkRequest b = baseRequest();
        b.rasterBackfaceCulling = false;
        REQUIRE_FALSE(a.sameParams(b));
    }
}

TEST_CASE("VDPM identity handles pack index + generation", "[vdpm][gpu-registry]")
{
    const VdpmFrontHandle f = makeHandle<VdpmFrontHandle>(7, 3);
    REQUIRE(handleIndex(f) == 7u);
    REQUIRE(handleGeneration(f) == 3u);
    REQUIRE(f != NullVdpmFront);

    const VdpmMeshHandle m = makeHandle<VdpmMeshHandle>(1234, 5);
    REQUIRE(handleIndex(m) == 1234u);
    REQUIRE(handleGeneration(m) == 5u);
    REQUIRE(m != NullVdpmMesh);
}

TEST_CASE("selectVisibleVdpmRequests filters to camera-visible fronts", "[vdpm][gpu-registry]")
{
    const VdpmFrontHandle fa = makeHandle<VdpmFrontHandle>(1, 1);
    const VdpmFrontHandle fb = makeHandle<VdpmFrontHandle>(2, 1);
    const VdpmFrontHandle fc = makeHandle<VdpmFrontHandle>(3, 1);

    VdpmWorkRequest ra = baseRequest();
    ra.front = fa;
    VdpmWorkRequest rb = baseRequest();
    rb.front = fb;
    VdpmWorkRequest rc = baseRequest();
    rc.front = fc;
    const std::array<VdpmWorkRequest, 3> sink{ra, rb, rc};

    // Only fa + fc are camera-visible; fb (shadow-only) is dropped. First-seen order preserved.
    const std::array<VdpmFrontHandle, 2> visible{fa, fc};
    std::vector<VdpmWorkRequest> kept;
    VdpmRequestSelectScratch scratch;
    selectVisibleVdpmRequests(sink, visible, kept, scratch);
    REQUIRE(kept.size() == 2);
    REQUIRE(kept[0].front == fa);
    REQUIRE(kept[1].front == fc);

    // Reusing the same scratch + output across calls clears them (retaining capacity), never
    // accumulates.
    const std::array<VdpmFrontHandle, 1> visibleC{fc};
    selectVisibleVdpmRequests(sink, visibleC, kept, scratch);
    REQUIRE(kept.size() == 1);
    REQUIRE(kept[0].front == fc);
}

TEST_CASE("selectVisibleVdpmRequests collapses an identical duplicate front",
          "[vdpm][gpu-registry]")
{
    const VdpmFrontHandle f = makeHandle<VdpmFrontHandle>(5, 1);
    VdpmWorkRequest r = baseRequest();
    r.front = f;
    // Same front twice with identical params (a front rendered into two forward passes) → one
    // entry.
    const std::array<VdpmWorkRequest, 2> sink{r, r};
    const std::array<VdpmFrontHandle, 1> visible{f};
    std::vector<VdpmWorkRequest> kept;
    VdpmRequestSelectScratch scratch;
    selectVisibleVdpmRequests(sink, visible, kept, scratch);
    REQUIRE(kept.size() == 1);
    REQUIRE(kept[0].front == f);
}

TEST_CASE("selectVisibleVdpmRequests throws on conflicting duplicate front", "[vdpm][gpu-registry]")
{
    const VdpmFrontHandle f = makeHandle<VdpmFrontHandle>(5, 1);
    VdpmWorkRequest r0 = baseRequest();
    r0.front = f;
    VdpmWorkRequest r1 = baseRequest();
    r1.front = f;
    r1.world[0, 3] = 9.0f; // same front, DIFFERENT params → conflict
    const std::array<VdpmWorkRequest, 2> sink{r0, r1};
    const std::array<VdpmFrontHandle, 1> visible{f};
    std::vector<VdpmWorkRequest> kept;
    VdpmRequestSelectScratch scratch;
    REQUIRE_THROWS_AS(selectVisibleVdpmRequests(sink, visible, kept, scratch), std::logic_error);
}

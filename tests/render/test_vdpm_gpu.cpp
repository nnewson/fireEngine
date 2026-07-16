#include <catch2/catch_test_macros.hpp>

#include <array>

#include <fire_engine/graphics/vdpm.hpp>
#include <fire_engine/math/mat4.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/render/device.hpp>
#include <fire_engine/render/vdpm_gpu.hpp>

using namespace fire_engine;

// GPU VDPM harness (rendering-spine #3, GPU-driven-front Stage B1). Tagged [.][gpu] so normal CTest
// (and the no-ICD CI runners) SKIP it — it needs a real Vulkan device. Run locally with:
//   ./test_fire_engine "[gpu]"
// from the build dir. It cross-checks the GPU compute port against the pure CPU scoring authority.

TEST_CASE("VDPM GPU: a surface-free compute device initialises", "[.][gpu]")
{
    // The one genuine unknown for Stage B1: MoltenVK offscreen compute with no swapchain. If this
    // throws, diagnose THAT failure (no hidden-window fallback) — MoltenVK supports compute without
    // a surface.
    const Device device = Device::headlessCompute();
    CHECK(*device.device() != nullptr);
    CHECK(*device.physicalDevice() != nullptr);
    CHECK(*device.graphicsQueue() != nullptr); // the shared graphics+compute queue

    // No surface + no present queue in headless mode.
    CHECK(*device.surface() == nullptr);
    CHECK(*device.presentQueue() == nullptr);

    // The selected family really is graphics+compute (the same-queue production path).
    const auto families = device.physicalDevice().getQueueFamilyProperties();
    REQUIRE(device.graphicsFamily() < families.size());
    const vk::QueueFlags flags = families[device.graphicsFamily()].queueFlags;
    CHECK(static_cast<bool>(flags & vk::QueueFlagBits::eGraphics));
    CHECK(static_cast<bool>(flags & vk::QueueFlagBits::eCompute));
}

// The ABI pack helpers are pure CPU field copies, so they run in NORMAL CI ([vdpm], not [gpu]) —
// no device needed. They pin the uploader faithful to the scoring authority, the third leg of the
// "one scoring" contract (oracle / uploader / shader).

TEST_CASE("packVdpmSplit copies the split's cone + errors + IDs", "[vdpm]")
{
    VertexSplit s;
    s.parent = 7;
    s.child = 42;
    s.normalConeAxis = Vec3{0.0f, 1.0f, 0.0f};
    s.normalConeCos = 0.25f;
    s.supportRadius = 1.5f;
    s.error = 2.0f;
    s.uvError = 3.0f;
    s.normalError = 0.5f;
    s.tangentError = 0.75f;

    const VdpmSplitGpu g = packVdpmSplit(s);
    CHECK(g.coneAxisCos[0] == 0.0f);
    CHECK(g.coneAxisCos[1] == 1.0f);
    CHECK(g.coneAxisCos[2] == 0.0f);
    CHECK(g.coneAxisCos[3] == 0.25f); // cos packed into .w
    CHECK(g.supportRadius == 1.5f);
    CHECK(g.error == 2.0f);
    CHECK(g.uvError == 3.0f);
    CHECK(g.normalError == 0.5f);
    CHECK(g.tangentError == 0.75f);
    CHECK(g.parentId == 7u);
    CHECK(g.childId == 42u);
}

TEST_CASE("packVdpmPosition packs xyz into a padded vec4", "[vdpm]")
{
    const VdpmPositionGpu g = packVdpmPosition(Vec3{1.0f, -2.0f, 3.0f});
    CHECK(g.position[0] == 1.0f);
    CHECK(g.position[1] == -2.0f);
    CHECK(g.position[2] == 3.0f);
    CHECK(g.position[3] == 0.0f);
}

TEST_CASE("packVdpmScoreParams images VdpmViewParams column-major with flags + addresses", "[vdpm]")
{
    // A FULLY NON-SYMMETRIC linear part with nine distinct entries, so a row/column transpose in the
    // packer would be caught (a symmetric or diagonal matrix could not). Invertible (det = -3) so
    // the cone stays usable. Translated so the affine term is distinct too.
    Mat4 world = Mat4::identity();
    world[0, 0] = 1.0f;
    world[0, 1] = 2.0f;
    world[0, 2] = 3.0f;
    world[1, 0] = 4.0f;
    world[1, 1] = 5.0f;
    world[1, 2] = 6.0f;
    world[2, 0] = 7.0f;
    world[2, 1] = 8.0f;
    world[2, 2] = 10.0f;
    world[0, 3] = 10.0f;
    world[1, 3] = 20.0f;
    world[2, 3] = 30.0f;
    const Vec3 cam{1.0f, 2.0f, 3.0f};
    const VdpmViewParams v =
        makeVdpmViewParams(world, cam, 1.25f, 1000.0f, 2.0f, true, 0.5f, 0.6f, 0.7f);

    const VdpmScoreParams p = packVdpmScoreParams(v, 0x1111u, 0x2222u, 0x3333u, 99u);

    // GLSL mat3 is column-major: packed column c, row r == worldLinear[r, c]. Assert all NINE
    // entries (transpose-detecting) plus all THREE padding slots.
    const std::array<const float*, 3> cols{p.worldLinearCol0, p.worldLinearCol1, p.worldLinearCol2};
    for (int c = 0; c < 3; ++c)
    {
        CHECK(cols[c][0] == v.worldLinear[0, c]);
        CHECK(cols[c][1] == v.worldLinear[1, c]);
        CHECK(cols[c][2] == v.worldLinear[2, c]);
        CHECK(cols[c][3] == 0.0f); // column padding
    }

    CHECK(p.worldTranslationMinusCamera[0] == v.worldTranslationMinusCamera.x());
    CHECK(p.worldTranslationMinusCamera[1] == v.worldTranslationMinusCamera.y());
    CHECK(p.worldTranslationMinusCamera[2] == v.worldTranslationMinusCamera.z());
    CHECK(p.cameraObj[0] == v.cameraObj.x());

    CHECK(p.worldLengthScale == v.worldLengthScale);
    CHECK(p.facingSign == v.facingSign);
    CHECK(p.projScaleY == v.projScaleY);
    CHECK(p.halfViewport == v.halfViewport); // == viewportHeight/2 = 500
    CHECK(p.silhouetteBoost == v.silhouetteBoost);
    CHECK(p.uvScale == v.uvScale);
    CHECK(p.normalScale == v.normalScale);
    CHECK(p.tangentScale == v.tangentScale);

    CHECK(p.coneUsable == 1u); // flags as uint32, not bool
    CHECK(p.coneCullEnabled == 1u);
    CHECK(p.splitCount == 99u);
    CHECK(p.splitsAddress == 0x1111u);
    CHECK(p.positionsAddress == 0x2222u);
    CHECK(p.outputsAddress == 0x3333u);
}

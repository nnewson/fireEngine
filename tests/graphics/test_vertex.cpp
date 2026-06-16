#include <fire_engine/graphics/vertex.hpp>

#include <support/test_traits.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using fire_engine::Colour3;
using fire_engine::Joints4;
using fire_engine::Vec2;
using fire_engine::Vec3;
using fire_engine::Vec4;
using fire_engine::Vertex;

// ==========================================================================
// Construction
// ==========================================================================

TEST_CASE("VertexConstruction.DefaultIsZeroed", "[VertexConstruction]")
{
    Vertex v;
    CHECK(v.position().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.position().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.position().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.colour().r() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.colour().g() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.colour().b() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.texCoord().s() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.texCoord().t() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.tangent().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.tangent().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.tangent().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.tangent().w() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.weights().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.weights().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.weights().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.weights().w() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.joints().j0() == 0u);
    CHECK(v.joints().j1() == 0u);
    CHECK(v.joints().j2() == 0u);
    CHECK(v.joints().j3() == 0u);
}

TEST_CASE("VertexConstruction.FullConstructor", "[VertexConstruction]")
{
    Vertex v({1.0f, 2.0f, 3.0f}, {0.5f, 0.6f, 0.7f}, {0.0f, 1.0f, 0.0f}, {0.25f, 0.75f});
    CHECK(v.position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(v.position().y() == Catch::Approx(2.0f).margin(1e-5f));
    CHECK(v.position().z() == Catch::Approx(3.0f).margin(1e-5f));
    CHECK(v.colour().r() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(v.colour().g() == Catch::Approx(0.6f).margin(1e-5f));
    CHECK(v.colour().b() == Catch::Approx(0.7f).margin(1e-5f));
    CHECK(v.normal().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().y() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(v.normal().z() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.texCoord().s() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(v.texCoord().t() == Catch::Approx(0.75f).margin(1e-5f));
}

TEST_CASE("VertexConstruction.FullConstructorWithTangent", "[VertexConstruction]")
{
    Vertex v({1.0f, 2.0f, 3.0f}, {0.5f, 0.6f, 0.7f}, {0.0f, 1.0f, 0.0f}, {0.25f, 0.75f}, {},
             {0.0f, 0.0f, 0.0f, 0.0f}, {0.1f, 0.2f, 0.3f, 1.0f});
    CHECK(v.tangent().x() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(v.tangent().y() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(v.tangent().z() == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(v.tangent().w() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("VertexConstruction.TangentHandednessNegative", "[VertexConstruction]")
{
    Vertex v({0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0}, {}, {0.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, -1.0f});
    CHECK(v.tangent().w() == Catch::Approx(-1.0f).margin(1e-5f));
}

TEST_CASE("VertexConstruction.WeightsConstructor", "[VertexConstruction]")
{
    Vertex v({0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0}, {1, 2, 3, 4}, {0.4f, 0.3f, 0.2f, 0.1f});
    CHECK(v.weights().x() == Catch::Approx(0.4f).margin(1e-5f));
    CHECK(v.weights().y() == Catch::Approx(0.3f).margin(1e-5f));
    CHECK(v.weights().z() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(v.weights().w() == Catch::Approx(0.1f).margin(1e-5f));
}

// ==========================================================================
// Accessors
// ==========================================================================

TEST_CASE("VertexAccessors.SetPosition", "[VertexAccessors]")
{
    Vertex v;
    v.position({4.0f, 5.0f, 6.0f});
    CHECK(v.position().x() == Catch::Approx(4.0f).margin(1e-5f));
    CHECK(v.position().y() == Catch::Approx(5.0f).margin(1e-5f));
    CHECK(v.position().z() == Catch::Approx(6.0f).margin(1e-5f));
}

TEST_CASE("VertexAccessors.SetColour", "[VertexAccessors]")
{
    Vertex v;
    v.colour({0.1f, 0.2f, 0.3f});
    CHECK(v.colour().r() == Catch::Approx(0.1f).margin(1e-5f));
    CHECK(v.colour().g() == Catch::Approx(0.2f).margin(1e-5f));
    CHECK(v.colour().b() == Catch::Approx(0.3f).margin(1e-5f));
}

TEST_CASE("VertexAccessors.SetNormal", "[VertexAccessors]")
{
    Vertex v;
    v.normal({0.0f, 0.0f, 1.0f});
    CHECK(v.normal().x() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().y() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.normal().z() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("VertexAccessors.SetTexCoords", "[VertexAccessors]")
{
    Vertex v;
    v.texCoord({0.5f, 0.75f});
    CHECK(v.texCoord().s() == Catch::Approx(0.5f).margin(1e-5f));
    CHECK(v.texCoord().t() == Catch::Approx(0.75f).margin(1e-5f));
}

TEST_CASE("VertexAccessors.SetTangent", "[VertexAccessors]")
{
    Vertex v;
    v.tangent({0.6f, 0.7f, 0.8f, -1.0f});
    CHECK(v.tangent().x() == Catch::Approx(0.6f).margin(1e-5f));
    CHECK(v.tangent().y() == Catch::Approx(0.7f).margin(1e-5f));
    CHECK(v.tangent().z() == Catch::Approx(0.8f).margin(1e-5f));
    CHECK(v.tangent().w() == Catch::Approx(-1.0f).margin(1e-5f));
}

TEST_CASE("VertexAccessors.SetJoints", "[VertexAccessors]")
{
    Vertex v;
    v.joints({10, 20, 30, 40});
    CHECK(v.joints().j0() == 10u);
    CHECK(v.joints().j1() == 20u);
    CHECK(v.joints().j2() == 30u);
    CHECK(v.joints().j3() == 40u);
}

TEST_CASE("VertexAccessors.SetWeights", "[VertexAccessors]")
{
    Vertex v;
    v.weights({0.25f, 0.25f, 0.25f, 0.25f});
    CHECK(v.weights().x() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(v.weights().y() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(v.weights().z() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(v.weights().w() == Catch::Approx(0.25f).margin(1e-5f));
}

// ==========================================================================
// Equality
// ==========================================================================

TEST_CASE("VertexEquality.IdenticalVertices", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.1f, 0.2f});
    Vertex b({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.1f, 0.2f});
    CHECK(a == b);
}

TEST_CASE("VertexEquality.DifferentPosition", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    Vertex b({9.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    CHECK_FALSE(a == b);
}

TEST_CASE("VertexEquality.DifferentColour", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    Vertex b({1.0f, 2.0f, 3.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    CHECK_FALSE(a == b);
}

TEST_CASE("VertexEquality.DifferentNormal", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    Vertex b({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f});
    CHECK_FALSE(a == b);
}

TEST_CASE("VertexEquality.DifferentTexS", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    Vertex b({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f});
    CHECK_FALSE(a == b);
}

TEST_CASE("VertexEquality.DifferentTexT", "[VertexEquality]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f});
    Vertex b({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f});
    CHECK_FALSE(a == b);
}

TEST_CASE("VertexEquality.DefaultVerticesAreEqual", "[VertexEquality]")
{
    Vertex a;
    Vertex b;
    CHECK(a == b);
}

// ==========================================================================
// Second UV set (TEXCOORD_1) — defaults to (0, 0); round-trips via accessors.
// ==========================================================================

TEST_CASE("VertexTexCoord1.DefaultsToZero", "[VertexTexCoord1]")
{
    Vertex v;
    CHECK(v.texCoord1().s() == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(v.texCoord1().t() == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("VertexTexCoord1.SetterRoundTrips", "[VertexTexCoord1]")
{
    Vertex v;
    v.texCoord1({0.25f, 0.75f});
    CHECK(v.texCoord1().s() == Catch::Approx(0.25f).margin(1e-5f));
    CHECK(v.texCoord1().t() == Catch::Approx(0.75f).margin(1e-5f));
}

// ==========================================================================
// Copy and Move Semantics
// ==========================================================================

TEST_CASE("VertexCopy.CopyConstructCreatesIndependentCopy", "[VertexCopy]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.1f, 0.2f});
    Vertex b{a};
    CHECK(a == b);

    b.position({9.0f, 9.0f, 9.0f});
    CHECK(a.position().x() == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("VertexCopy.CopyAssignCreatesIndependentCopy", "[VertexCopy]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.1f, 0.2f});
    Vertex b;
    b = a;
    CHECK(a == b);
}

TEST_CASE("VertexMove.MoveConstructTransfersState", "[VertexMove]")
{
    Vertex a({1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.1f, 0.2f});
    Vertex b{std::move(a)};
    CHECK(b.position().x() == Catch::Approx(1.0f).margin(1e-5f));
    CHECK(b.texCoord().s() == Catch::Approx(0.1f).margin(1e-5f));
}

// ==========================================================================
// Noexcept guarantees
// ==========================================================================

TEST_CASE("VertexNoexcept.AllAccessorsAreNoexcept", "[VertexNoexcept]")
{
    static_assert(std::is_nothrow_default_constructible_v<Vertex>);
    static_assert(test_traits::nothrow_constructible_from_v<Vertex, Vec3, Colour3, Vec3, Vec2>);
    static_assert(
        test_traits::has_nothrow_vertex_accessors<Vertex, Vec2, Vec3, Vec4, Colour3, Joints4>);
    static_assert(test_traits::has_nothrow_equality<Vertex>);
}

// ==========================================================================
// Layout — Vulkan vertex input compatibility
// ==========================================================================

TEST_CASE("VertexLayout.SizeAndOffsets", "[VertexLayout]")
{
    // Vec2 and Vec4 are standard-layout with float data_[N]
    static_assert(sizeof(Vec2) == sizeof(float) * 2);
    static_assert(sizeof(Vec4) == sizeof(float) * 4);

    // Vertex size: position(Vec3 12) + colour(Colour3 12) + normal(Vec3 12)
    //            + texCoord(Vec2 8) + joints(Joints4 16) + weights(Vec4 16)
    //            + tangent(Vec4 16) + texCoord1(Vec2 8) = 100 bytes
    static_assert(sizeof(Joints4) == sizeof(uint32_t) * 4);
    static_assert(sizeof(Vertex) == sizeof(Vec3) * 2 + sizeof(Colour3) + sizeof(Vec2) * 2 +
                                        sizeof(Joints4) + sizeof(Vec4) * 2);
}

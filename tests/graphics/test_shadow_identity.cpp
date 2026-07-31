#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fire_engine/core/node_id.hpp>
#include <fire_engine/graphics/lighting.hpp>
#include <fire_engine/graphics/shadow_identity.hpp>
#include <fire_engine/scene/light.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/scene_graph.hpp>

using namespace fire_engine;

// ---------------------------------------------------------------------------
// Identities for shadow-LOD hysteresis (SH-03).
//
// Hysteresis is state carried BETWEEN frames, so its key must name the same thing next frame as it
// did last frame. Everything here exists because the cheap alternatives alias: a physical slot is
// reassigned when the active light set changes, an array index shifts when a node is added, and a
// pointer can be recycled. Each such alias applies one caster's or one light's history to a
// different one, with no symptom beyond shadows behaving oddly near thresholds.
// ---------------------------------------------------------------------------

TEST_CASE("ShadowIdentity.NodeIdsAreUniqueAndNeverTheInvalidSentinel", "[ShadowIdentity]")
{
    std::unordered_set<NodeId> seen;
    for (int i = 0; i < 64; ++i)
    {
        const Node node{"light"};
        CHECK(node.id() != NodeId::Invalid);
        CHECK(seen.insert(node.id()).second); // never reused, even though each node is destroyed
    }
}

TEST_CASE("ShadowIdentity.MoveConstructionLeavesTwoDistinctLiveIdentities", "[ShadowIdentity]")
{
    // The failure a defaulted move produces: the scalar is copied, and BOTH the moved-from and
    // moved-to objects are alive holding one "process-unique" id. Checking only the destination
    // hides it entirely.
    Node source{"spot"};
    const NodeId original = source.id();
    const Node moved{std::move(source)};

    CHECK(moved.id() == original);         // identity follows the contents
    CHECK(source.id() != NodeId::Invalid); // NOLINT(bugprone-use-after-move): still a live object
    CHECK(source.id() != moved.id());      // NOLINT(bugprone-use-after-move): and distinct
}

TEST_CASE("ShadowIdentity.MoveAssignmentLeavesTwoDistinctLiveIdentities", "[ShadowIdentity]")
{
    Node source{"spot"};
    Node target{"point"};
    const NodeId original = source.id();
    target = std::move(source);

    CHECK(target.id() == original);
    CHECK(source.id() != NodeId::Invalid); // NOLINT(bugprone-use-after-move)
    CHECK(source.id() != target.id());     // NOLINT(bugprone-use-after-move)
}

TEST_CASE("ShadowIdentity.CasterIdsAreUniqueAndNeverTheInvalidSentinel", "[ShadowIdentity]")
{
    std::unordered_set<std::uint64_t> seen;
    for (int i = 0; i < 64; ++i)
    {
        const ShadowCasterId id = allocateShadowCasterId();
        CHECK(id != ShadowCasterId::Invalid);
        CHECK(seen.insert(static_cast<std::uint64_t>(id)).second);
    }
}

TEST_CASE("ShadowIdentity.WorldOnlySharesTheCascadeIdentity", "[ShadowIdentity]")
{
    // Load-bearing: the full and world-only cascades must make the SAME choice for a given rigid
    // caster. Sharing one key guarantees it, rather than hoping two independent computations agree.
    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        CHECK(ShadowLogicalViewId::worldOnly(cascade) == ShadowLogicalViewId::cascade(cascade));
    }
}

TEST_CASE("ShadowIdentity.DefaultConstructedViewsAreInvalidNotCascadeZero", "[ShadowIdentity]")
{
    // The trap a public aggregate would set: a default value that silently names a REAL view,
    // whose history it would then corrupt.
    const ShadowLogicalViewId defaulted;

    CHECK_FALSE(defaulted.valid());
    CHECK(defaulted.kind() == ShadowLogicalViewKind::Invalid);
    CHECK_FALSE(defaulted == ShadowLogicalViewId::cascade(0));
    CHECK_FALSE(ShadowLodStateKey{}.valid());
}

TEST_CASE("ShadowIdentity.FactoriesRejectOutOfDomainInput", "[ShadowIdentity]")
{
    // Every one of these would otherwise produce a key that looks valid and names the wrong view.
    CHECK_FALSE(ShadowLogicalViewId::cascade(kShadowCascadeCount).valid());
    CHECK_FALSE(ShadowLogicalViewId::self(0).valid()); // 0 = an object that was never loaded
    CHECK_FALSE(ShadowLogicalViewId::spot(NodeId::Invalid).valid());
    CHECK_FALSE(ShadowLogicalViewId::point(NodeId::Invalid, 0).valid());
    CHECK_FALSE(ShadowLogicalViewId::point(static_cast<NodeId>(7), kCubeFaceCount).valid());

    CHECK(ShadowLogicalViewId::cascade(kShadowCascadeCount - 1).valid());
    CHECK(ShadowLogicalViewId::point(static_cast<NodeId>(7), kCubeFaceCount - 1).valid());
}

TEST_CASE("ShadowIdentity.NonPointKindsCarryNoFace", "[ShadowIdentity]")
{
    CHECK(ShadowLogicalViewId::cascade(2).face() == 0);
    CHECK(ShadowLogicalViewId::self(5).face() == 0);
    CHECK(ShadowLogicalViewId::spot(static_cast<NodeId>(9)).face() == 0);
    CHECK(ShadowLogicalViewId::point(static_cast<NodeId>(9), 4).face() == 4);
}

TEST_CASE("ShadowIdentity.DistinctViewsDoNotShareAKey", "[ShadowIdentity]")
{
    const auto light = static_cast<NodeId>(42);
    const std::array<ShadowLogicalViewId, 7> views{
        ShadowLogicalViewId::cascade(0),     ShadowLogicalViewId::cascade(1),
        ShadowLogicalViewId::self(1),        ShadowLogicalViewId::self(2),
        ShadowLogicalViewId::spot(light),    ShadowLogicalViewId::point(light, 0),
        ShadowLogicalViewId::point(light, 1)};

    for (std::size_t i = 0; i < views.size(); ++i)
    {
        for (std::size_t j = i + 1; j < views.size(); ++j)
        {
            CAPTURE(i, j);
            CHECK_FALSE(views[i] == views[j]);
        }
    }
    // Same numeric value, different kinds. The spot/point pair is unreachable today (a light is
    // only ever one kind), but the key type does not lean on that invariant.
    CHECK_FALSE(ShadowLogicalViewId::cascade(3) == ShadowLogicalViewId::self(3));
    CHECK_FALSE(ShadowLogicalViewId::spot(light) == ShadowLogicalViewId::point(light, 0));
}

TEST_CASE("ShadowIdentity.AReplacedLodChainDoesNotFindOldHistory", "[ShadowIdentity]")
{
    // The generation is part of the KEY, so a caster whose shadow geometry was replaced simply
    // finds nothing — no call site has to remember to compare a revision field.
    const auto caster = static_cast<ShadowCasterId>(11);
    const auto view = ShadowLogicalViewId::cascade(1);
    const ShadowLodStateKey first{caster, ShadowCasterGeneration::First, view};
    const ShadowLodStateKey second{caster, static_cast<ShadowCasterGeneration>(1), view};

    std::unordered_map<ShadowLodStateKey, int> history;
    history[first] = 2; // "this caster held level 2 in cascade 1"

    CHECK_FALSE(first == second);
    CHECK(history.contains(first));
    CHECK_FALSE(history.contains(second));
}

TEST_CASE("ShadowIdentity.StateKeysSeparateCastersAndViews", "[ShadowIdentity]")
{
    const ShadowLodStateKey a{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                              ShadowLogicalViewId::cascade(0)};
    const ShadowLodStateKey b{static_cast<ShadowCasterId>(2), ShadowCasterGeneration::First,
                              ShadowLogicalViewId::cascade(0)};
    const ShadowLodStateKey c{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                              ShadowLogicalViewId::cascade(1)};
    const ShadowLodStateKey d{static_cast<ShadowCasterId>(2), ShadowCasterGeneration::First,
                              ShadowLogicalViewId::cascade(1)};

    std::unordered_map<ShadowLodStateKey, int> history;
    history[a] = 1;
    history[b] = 2;
    history[c] = 3;
    history[d] = 4;

    CHECK(history.size() == 4);
    CHECK(history.at(a) == 1);
    CHECK(history.at(d) == 4);
    // A freshly built equal key finds its own entry — the property the whole design turns on.
    CHECK(
        history.at(ShadowLodStateKey{static_cast<ShadowCasterId>(1), ShadowCasterGeneration::First,
                                     ShadowLogicalViewId::cascade(0)}) == 1);
}

TEST_CASE("ShadowIdentity.HashIsConsistentWithEquality", "[ShadowIdentity]")
{
    // The ACTUAL contract. An earlier version of this test asserted that two unequal keys hash
    // differently — which no finite hash can promise, and which would not matter if it failed:
    // equality decides membership, and colliding keys still coexist and retrieve independently.
    const ShadowLodStateKey lhs{static_cast<ShadowCasterId>(5), ShadowCasterGeneration::First,
                                ShadowLogicalViewId::self(7)};
    const ShadowLodStateKey same{static_cast<ShadowCasterId>(5), ShadowCasterGeneration::First,
                                 ShadowLogicalViewId::self(7)};
    const ShadowLodStateKey swapped{static_cast<ShadowCasterId>(7), ShadowCasterGeneration::First,
                                    ShadowLogicalViewId::self(5)};

    // Equal keys MUST hash equally, or a lookup can miss its own entry.
    CHECK(lhs == same);
    CHECK(std::hash<ShadowLodStateKey>{}(lhs) == std::hash<ShadowLodStateKey>{}(same));

    // Unequal keys coexist and retrieve independently, whatever their hashes do.
    std::unordered_map<ShadowLodStateKey, int> history;
    history[lhs] = 1;
    history[swapped] = 2;
    CHECK(history.size() == 2);
    CHECK(history.at(lhs) == 1);
    CHECK(history.at(swapped) == 2);
}

TEST_CASE("ShadowIdentity.GatheredLightsCarryTheirOwningNodeId", "[ShadowIdentity]")
{
    // The round trip that matters: a light's identity has to survive the gather, or the shadow
    // pass has nothing stable to key punctual hysteresis on.
    SceneGraph scene;
    auto spotNode = std::make_unique<Node>("Spot");
    spotNode->component().emplace<Light>().type(Light::Type::Spot);
    const NodeId spotId = spotNode->id();
    scene.addNode(std::move(spotNode));

    auto pointNode = std::make_unique<Node>("Point");
    pointNode->component().emplace<Light>().type(Light::Type::Point);
    const NodeId pointId = pointNode->id();
    scene.addNode(std::move(pointNode));

    scene.resolve();
    const std::vector<Lighting> lights = scene.gatherLights();

    REQUIRE(lights.size() == 2);
    CHECK(spotId != pointId);

    // ASSOCIATION, not mere presence: find each light by its TYPE and assert it carries its own
    // node's id. Checking only that both ids appear somewhere would pass just as happily with the
    // two swapped — which is the failure that matters, since a swapped id hands one light's
    // hysteresis history to another.
    const auto byType = [&lights](Light::Type type) -> const Lighting*
    {
        const auto match = std::ranges::find_if(lights, [&](const Lighting& l)
                                                { return l.type == std::to_underlying(type); });
        return match == lights.end() ? nullptr : &*match;
    };

    const Lighting* spotLight = byType(Light::Type::Spot);
    const Lighting* pointLight = byType(Light::Type::Point);
    REQUIRE(spotLight != nullptr);
    REQUIRE(pointLight != nullptr);
    CHECK(spotLight->nodeId == spotId);
    CHECK(pointLight->nodeId == pointId);
}

TEST_CASE("ShadowIdentity.GenerationAdvancesAndRefusesToWrap", "[ShadowIdentity]")
{
    // No production caller today (SH-04 removed the shadow-proxy setter), so without this the
    // advancer would be unexercised code waiting for the proxy API to trust it blindly.
    CHECK(nextShadowCasterGeneration(ShadowCasterGeneration::First) ==
          static_cast<ShadowCasterGeneration>(1));
    CHECK(nextShadowCasterGeneration(static_cast<ShadowCasterGeneration>(41)) ==
          static_cast<ShadowCasterGeneration>(42));

    // Exhaustion is deliberately NOT covered here: at UINT64_MAX the advancer logs and calls
    // std::abort() rather than wrapping to First, where a caster would find history recorded
    // against a chain replaced 2^64 generations ago. A test cannot survive that call, and weakening
    // it to a sentinel return so it could be asserted would trade a loud, unreachable failure for a
    // quiet, reachable one.
}

#pragma once

#include <cstddef>
#include <vector>

namespace fire_engine
{

// A simulation island: a connected component of dynamic bodies linked by contacts
// and/or joints. It is the unit of the per-island solve (each island is solved
// independently) and, in P5.2, of sleeping. Members are indices into the caller's
// arrays — `bodies` into the body/SolverBody array, `contacts`/`joints` into the
// contact-input / joint-input arrays handed to the solver.
struct Island
{
    std::vector<int> bodies;
    std::vector<int> contacts;
    std::vector<int> joints;
};

// One connectivity edge for island construction: the two body indices a contact or
// a joint couples. A -1 endpoint (or a non-dynamic one) is a boundary — it never
// merges two islands, but the edge is still attributed to its dynamic endpoint.
struct IslandEdge
{
    int bodyA{-1};
    int bodyB{-1};
};

// Partition the movable bodies into islands via union-find. `movable[i]` marks body
// `i` as an island node — Dynamic or Kinematic, both of which the solver can move (a
// Kinematic is position-corrected out of penetration). Only Static bodies are
// boundaries: they appear in edges but never connect two islands, so a box on the
// floor isn't joined to another box through the floor. An edge whose endpoints are
// both movable merges their islands; every movable body is in exactly one island (a
// singleton if isolated). Each contact/joint is attributed to its movable endpoint's
// island. (Kinematic nodes own their contacts so they get solved, but the caller
// integrates velocity only for Dynamic members.)
//
// Deterministic: islands are returned ordered by their lowest member body index,
// members ascending, contacts/joints in input order — no hash-map iteration leaks
// into the result.
[[nodiscard]]
std::vector<Island> buildIslands(std::size_t bodyCount, const std::vector<bool>& movable,
                                 const std::vector<IslandEdge>& contactEdges,
                                 const std::vector<IslandEdge>& jointEdges);

} // namespace fire_engine

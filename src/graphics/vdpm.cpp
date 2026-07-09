#include <fire_engine/graphics/vdpm.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace fire_engine
{

namespace
{

// Exact-position weld key — identical to the simplifier's: glTF splits a corner into a render wedge
// per attribute seam, and those all weld to one canonical vertex so the collapse topology isn't
// shattered. Matching the simplifier's weld makes the recorded collapses' canonical kept/removed
// indices line up with the adjacency built here.
struct PosKey
{
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    bool operator==(const PosKey&) const noexcept = default;
};

struct PosKeyHash
{
    std::size_t operator()(const PosKey& k) const noexcept
    {
        std::size_t h = k.x;
        h = h * 1000003u ^ k.y;
        h = h * 1000003u ^ k.z;
        return h;
    }
};

[[nodiscard]] PosKey posKey(const Vec3& p) noexcept
{
    return PosKey{std::bit_cast<std::uint32_t>(p.x()), std::bit_cast<std::uint32_t>(p.y()),
                  std::bit_cast<std::uint32_t>(p.z())};
}

// original vertex -> canonical (first vertex at each exact position), matching the simplifier's
// weld.
[[nodiscard]] std::vector<std::uint32_t> weldVertices(std::span<const Vertex> vertices)
{
    const auto n = static_cast<std::uint32_t>(vertices.size());
    std::vector<std::uint32_t> weld(n);
    std::unordered_map<PosKey, std::uint32_t, PosKeyHash> weldMap;
    weldMap.reserve(n);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        weld[v] = weldMap.try_emplace(posKey(vertices[v].position()), v).first->second;
    }
    return weld;
}

// The finest triangle set in canonical space: each input triangle welded, degenerate (post-weld
// duplicate-vertex) triangles dropped.
[[nodiscard]] std::vector<std::array<std::uint32_t, 3>>
canonicalFaces(std::span<const std::uint32_t> weld, std::span<const std::uint32_t> indices)
{
    std::vector<std::array<std::uint32_t, 3>> faces;
    faces.reserve(indices.size() / 3);
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const std::array<std::uint32_t, 3> f{weld[indices[i]], weld[indices[i + 1]],
                                             weld[indices[i + 2]]};
        if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
        {
            continue;
        }
        faces.push_back(f);
    }
    return faces;
}

} // namespace

VertexForest buildVertexForest(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                               std::span<const MeshCollapse> collapses)
{
    const auto n = static_cast<std::uint32_t>(vertices.size());

    VertexForest forest;
    forest.vertexCount = vertices.size();
    forest.removingSplit.assign(n, kNoSplit);
    forest.splits.reserve(collapses.size());

    const std::vector<std::uint32_t> weld = weldVertices(vertices);
    std::vector<std::array<std::uint32_t, 3>> faces = canonicalFaces(weld, indices);
    std::vector<bool> faceLive(faces.size(), true);

    // Per-vertex incident live faces. Faces are remapped in place as the stream replays, so
    // faces[fi] always holds the current canonical reps and vertexFaces[v] the live faces currently
    // incident to canonical vertex v (stale non-live entries are simply skipped, never compacted).
    std::vector<std::vector<std::uint32_t>> vertexFaces(n);
    for (std::uint32_t fi = 0; fi < faces.size(); ++fi)
    {
        for (const std::uint32_t w : faces[fi])
        {
            vertexFaces[w].push_back(fi);
        }
    }

    auto faceHas = [&faces](std::uint32_t fi, std::uint32_t v)
    { return faces[fi][0] == v || faces[fi][1] == v || faces[fi][2] == v; };
    auto apexOf = [&faces](std::uint32_t fi, std::uint32_t u, std::uint32_t v)
    {
        for (const std::uint32_t w : faces[fi])
        {
            if (w != u && w != v)
            {
                return w;
            }
        }
        return kInvalidVertex;
    };

    for (const MeshCollapse& c : collapses)
    {
        const std::uint32_t kept = c.kept;
        const std::uint32_t removed = c.removed;

        // The collapsing edge's live faces = faces incident to `removed` that also contain `kept`.
        // Exactly one (boundary -> vl only) or two (interior -> the two opposite apexes vl/vr).
        std::uint32_t vl = kInvalidVertex;
        std::uint32_t vr = kInvalidVertex;
        int edgeFaceCount = 0;
        bool manifold = true;
        for (const std::uint32_t fi : vertexFaces[removed])
        {
            if (!faceLive[fi] || !faceHas(fi, kept))
            {
                continue;
            }
            const std::uint32_t apex = apexOf(fi, kept, removed);
            if (edgeFaceCount == 0)
            {
                vl = apex;
            }
            else if (edgeFaceCount == 1)
            {
                vr = apex;
            }
            else
            {
                manifold = false; // > 2 faces on the edge — non-manifold
            }
            ++edgeFaceCount;
        }

        // 0 faces = the stream diverged from this topology view; > 2 = non-manifold. Both are
        // unsupported: skip recording a split (the vertex stays never-removed = always active, a
        // conservative fallback) rather than emit a corrupt dependency.
        if (edgeFaceCount == 0 || !manifold)
        {
            continue;
        }

        const auto splitIndex = static_cast<std::uint32_t>(forest.splits.size());
        forest.splits.push_back(VertexSplit{kept, removed, vl, vr, c.error});
        forest.removingSplit[removed] = splitIndex;

        // Apply the collapse: retire the edge's faces, then remap `removed` -> `kept` in every
        // other face still incident to `removed`, moving those faces onto `kept`.
        for (const std::uint32_t fi : vertexFaces[removed])
        {
            if (!faceLive[fi])
            {
                continue;
            }
            if (faceHas(fi, kept))
            {
                faceLive[fi] = false; // one of the collapsed edge's faces
                continue;
            }
            for (std::uint32_t& w : faces[fi])
            {
                if (w == removed)
                {
                    w = kept;
                }
            }
            vertexFaces[kept].push_back(fi);
        }
        vertexFaces[removed].clear();
    }

    return forest;
}

ActiveFront ActiveFront::build(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                               std::span<const MeshCollapse> collapses)
{
    ActiveFront front;
    front.forest_ = buildVertexForest(vertices, indices, collapses);
    front.finestFaces_ = canonicalFaces(weldVertices(vertices), indices);

    const std::size_t n = vertices.size();
    // Coarsest state: only never-removed (root) canonical vertices are active; no split refined.
    front.active_.assign(n, false);
    for (std::uint32_t v = 0; v < n; ++v)
    {
        if (front.forest_.removingSplit[v] == kNoSplit)
        {
            front.active_[v] = true;
        }
    }
    front.refined_.assign(front.forest_.splits.size(), false);
    front.dependents_.assign(n, 0);
    return front;
}

std::uint32_t ActiveFront::activeAncestor(std::uint32_t canonicalVertex) const
{
    // A root is always active and has removingSplit == kNoSplit, so an inactive vertex always has a
    // valid removing split whose parent is one step nearer an active ancestor.
    std::uint32_t v = canonicalVertex;
    while (!active_[v])
    {
        v = forest_.splits[forest_.removingSplit[v]].parent;
    }
    return v;
}

bool ActiveFront::refine(std::uint32_t splitIndex)
{
    if (splitIndex >= forest_.splits.size() || refined_[splitIndex])
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (!active_[s.parent] || !active_[s.vl] || (s.vr != kInvalidVertex && !active_[s.vr]))
    {
        return false;
    }
    refined_[splitIndex] = true;
    active_[s.child] = true;
    ++dependents_[s.parent];
    ++dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        ++dependents_[s.vr];
    }
    return true;
}

bool ActiveFront::coarsen(std::uint32_t splitIndex)
{
    if (splitIndex >= forest_.splits.size() || !refined_[splitIndex])
    {
        return false;
    }
    const VertexSplit& s = forest_.splits[splitIndex];
    if (dependents_[s.child] != 0)
    {
        return false; // the child props up a refined split — not a leaf
    }
    refined_[splitIndex] = false;
    active_[s.child] = false;
    --dependents_[s.parent];
    --dependents_[s.vl];
    if (s.vr != kInvalidVertex)
    {
        --dependents_[s.vr];
    }
    return true;
}

void ActiveFront::refineAll()
{
    // Reverse stream order: a split's parent/vl/vr are removed only by later collapses, so undoing
    // coarsest-first guarantees each split's dependencies are already active.
    for (std::uint32_t i = static_cast<std::uint32_t>(forest_.splits.size()); i-- > 0;)
    {
        refine(i);
    }
}

void ActiveFront::coarsenAll()
{
    // Fixpoint: repeatedly collapse every currently-leaf refined split until none remain.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::uint32_t i = 0; i < forest_.splits.size(); ++i)
        {
            if (refined_[i] && coarsen(i))
            {
                changed = true;
            }
        }
    }
}

std::vector<std::array<std::uint32_t, 3>> ActiveFront::emitActiveCanonical() const
{
    std::vector<std::array<std::uint32_t, 3>> out;
    out.reserve(finestFaces_.size());
    for (const std::array<std::uint32_t, 3>& f : finestFaces_)
    {
        const std::array<std::uint32_t, 3> a{activeAncestor(f[0]), activeAncestor(f[1]),
                                             activeAncestor(f[2])};
        if (a[0] == a[1] || a[1] == a[2] || a[0] == a[2])
        {
            continue; // the face collapsed to a degenerate at the current front
        }
        out.push_back(a);
    }
    return out;
}

} // namespace fire_engine

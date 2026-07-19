// Pure VDPM repair face classifier — shared by vdpm_repair_detect.comp (the per-face detect dispatch)
// and the persistent repair kernel. It does NO buffer access, NO atomics, and NO required/control
// marking: the caller fetches the face/front data (corner + ancestor world positions, per-corner
// active flags, ancestor-equality degeneracy) and the frame params, calls classifyRepairFace, then
// APPLIES the result (marks the removing splits) in its own execution model. This lets the two
// callers share the exact screen-space policy — a faithful port of CPU detail::isFoldover +
// detail::classifyCoverageRepair (graphics/vdpm.cpp) — without coupling their dispatch shapes. All
// positions passed in are WORLD space; `viewProj` is jitter-free.
#ifndef VDPM_REPAIR_CLASSIFY_GLSL
#define VDPM_REPAIR_CLASSIFY_GLSL

const float kRepairMinCoverageScreenAreaPx = 2.0f;

const uint kRepairCoverageNone = 0u;         // no coverage repair for this face
const uint kRepairCoverageAllInactive = 1u;  // refine every inactive corner
const uint kRepairCoverageWorst = 2u;        // refine the single worst-displaced inactive corner

struct RepairClassification
{
    bool foldover;      // world-winding flip of the replacement vs the original
    uint coverageKind;  // kRepairCoverage*
    uint worstLocal;    // 0..2, the corner to refine when coverageKind == Worst
};

float rc_edge(vec2 a, vec2 b, vec2 p)
{
    return ((p.x - a.x) * (b.y - a.y)) - ((p.y - a.y) * (b.x - a.x));
}

bool rc_insideTri(vec2 p, vec2 a, vec2 b, vec2 c)
{
    const float d0 = rc_edge(a, b, p);
    const float d1 = rc_edge(b, c, p);
    const float d2 = rc_edge(c, a, p);
    return !((d0 < 0.0 || d1 < 0.0 || d2 < 0.0) && (d0 > 0.0 || d1 > 0.0 || d2 > 0.0));
}

// NDC xy of a world point; false (leaving `ndc` untouched) when behind the camera (w <= eps).
bool rc_toNdc(mat4 viewProj, vec3 wp, out vec2 ndc)
{
    const vec4 c = viewProj * vec4(wp, 1.0);
    if (c.w <= 1e-6)
    {
        return false;
    }
    ndc = c.xy / c.w;
    return true;
}

// `active0/1/2` are the per-corner finest flags; `degenerate` = the caller's ancestor-id equality
// (a0==a1 || a1==a2 || a0==a2). `cull` = the material's raster back-face-culling policy.
RepairClassification classifyRepairFace(vec3 o0, vec3 o1, vec3 o2, vec3 r0, vec3 r1, vec3 r2,
                                        bool active0, bool active1, bool active2, bool degenerate,
                                        mat4 viewProj, vec3 cameraPos, vec2 viewportWH, bool cull)
{
    RepairClassification rc;
    rc.foldover = false;
    rc.coverageKind = kRepairCoverageNone;
    rc.worstLocal = 0u;

    // --- isFoldover: WORLD-space winding of replacement vs original (degenerate ⇒ not a foldover).
    if (!degenerate)
    {
        const vec3 origN = cross(o1 - o0, o2 - o0);
        const vec3 replN = cross(r1 - r0, r2 - r0);
        rc.foldover = dot(origN, replN) < 0.0;
    }

    // --- classifyCoverageRepair.
    const bool anyInactive = !active0 || !active1 || !active2;
    const vec3 faceCentroid = (o0 + o1 + o2) * (1.0 / 3.0);
    const vec3 gn = cross(o1 - o0, o2 - o0);
    bool coverageResolved = false;
    if (cull && dot(gn, cameraPos - faceCentroid) <= 0.0)
    {
        coverageResolved = true; // back-facing + culled ⇒ never shown, can't leak
    }

    vec2 s0;
    vec2 s1;
    vec2 s2;
    if (!coverageResolved)
    {
        const int inFront = int(rc_toNdc(viewProj, o0, s0)) + int(rc_toNdc(viewProj, o1, s1)) +
                            int(rc_toNdc(viewProj, o2, s2));
        if (inFront == 0)
        {
            coverageResolved = true; // fully behind the camera — not visible
        }
        else if (inFront < 3)
        {
            if (anyInactive)
            {
                rc.coverageKind = kRepairCoverageAllInactive; // straddles near plane — refine
            }
            coverageResolved = true;
        }
    }
    if (!coverageResolved)
    {
        const float minNdcArea =
            kRepairMinCoverageScreenAreaPx / max(1.0, 0.25 * viewportWH.x * viewportWH.y);
        if (abs(rc_edge(s0, s1, s2)) * 0.5 < minNdcArea)
        {
            coverageResolved = true; // sub-pixel: not a visible hole
        }
    }
    if (!coverageResolved && degenerate)
    {
        if (anyInactive)
        {
            rc.coverageKind = kRepairCoverageAllInactive; // degenerate over a visible area ⇒ hole
        }
        coverageResolved = true;
    }
    vec2 sc;
    vec2 sa0;
    vec2 sa1;
    vec2 sa2;
    if (!coverageResolved)
    {
        if (!rc_toNdc(viewProj, faceCentroid, sc) || !rc_toNdc(viewProj, r0, sa0) ||
            !rc_toNdc(viewProj, r1, sa1) || !rc_toNdc(viewProj, r2, sa2))
        {
            if (anyInactive)
            {
                rc.coverageKind = kRepairCoverageAllInactive; // replacement/centroid straddles near
            }
            coverageResolved = true;
        }
        else if (rc_insideTri(sc, sa0, sa1, sa2))
        {
            coverageResolved = true; // the replacement still covers this face's centroid
        }
    }
    if (!coverageResolved)
    {
        // Coverage hole: the inactive corner whose ancestor is displaced furthest on screen.
        float worstDisp = -1.0;
        uint worstLocal = 3u;
        for (uint k = 0u; k < 3u; ++k)
        {
            const bool activeK = (k == 0u) ? active0 : (k == 1u ? active1 : active2);
            if (activeK)
            {
                continue; // already at finest — cannot displace
            }
            const vec3 fine = (k == 0u) ? o0 : (k == 1u ? o1 : o2);
            const vec3 anc = (k == 0u) ? r0 : (k == 1u ? r1 : r2);
            vec2 sFine;
            vec2 sAnc;
            if (!rc_toNdc(viewProj, fine, sFine) || !rc_toNdc(viewProj, anc, sAnc))
            {
                continue;
            }
            const vec2 d = sFine - sAnc;
            const float disp = dot(d, d);
            if (disp > worstDisp)
            {
                worstDisp = disp;
                worstLocal = k;
            }
        }
        if (worstLocal != 3u)
        {
            rc.coverageKind = kRepairCoverageWorst;
            rc.worstLocal = worstLocal;
        }
    }
    return rc;
}

#endif // VDPM_REPAIR_CLASSIFY_GLSL

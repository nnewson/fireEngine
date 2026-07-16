#include <fire_engine/render/vdpm_gpu.hpp>

namespace fire_engine
{

VdpmSplitGpu packVdpmSplit(const VertexSplit& split) noexcept
{
    VdpmSplitGpu g;
    g.coneAxisCos[0] = split.normalConeAxis.x();
    g.coneAxisCos[1] = split.normalConeAxis.y();
    g.coneAxisCos[2] = split.normalConeAxis.z();
    g.coneAxisCos[3] = split.normalConeCos;
    g.supportRadius = split.supportRadius;
    g.error = split.error;
    g.uvError = split.uvError;
    g.normalError = split.normalError;
    g.tangentError = split.tangentError;
    g.parentId = split.parent;
    g.childId = split.child;
    return g;
}

VdpmPositionGpu packVdpmPosition(const Vec3& position) noexcept
{
    VdpmPositionGpu g;
    g.position[0] = position.x();
    g.position[1] = position.y();
    g.position[2] = position.z();
    g.position[3] = 0.0f;
    return g;
}

VdpmScoreParams packVdpmScoreParams(const VdpmViewParams& view, std::uint64_t splitsAddress,
                                    std::uint64_t positionsAddress, std::uint64_t outputsAddress,
                                    std::uint32_t splitCount) noexcept
{
    VdpmScoreParams p;
    // GLSL mat3 is column-major, matching Mat3::fromColumns: column c = (worldLinear[0,c],
    // worldLinear[1,c], worldLinear[2,c]). The w of each padded column is unused.
    for (int c = 0; c < 3; ++c)
    {
        float* col = c == 0 ? p.worldLinearCol0 : (c == 1 ? p.worldLinearCol1 : p.worldLinearCol2);
        col[0] = view.worldLinear[0, c];
        col[1] = view.worldLinear[1, c];
        col[2] = view.worldLinear[2, c];
        col[3] = 0.0f;
    }
    p.worldTranslationMinusCamera[0] = view.worldTranslationMinusCamera.x();
    p.worldTranslationMinusCamera[1] = view.worldTranslationMinusCamera.y();
    p.worldTranslationMinusCamera[2] = view.worldTranslationMinusCamera.z();
    p.cameraObj[0] = view.cameraObj.x();
    p.cameraObj[1] = view.cameraObj.y();
    p.cameraObj[2] = view.cameraObj.z();
    p.worldLengthScale = view.worldLengthScale;
    p.facingSign = view.facingSign;
    p.projScaleY = view.projScaleY;
    p.halfViewport = view.halfViewport;
    p.silhouetteBoost = view.silhouetteBoost;
    p.uvScale = view.uvScale;
    p.normalScale = view.normalScale;
    p.tangentScale = view.tangentScale;
    p.coneUsable = view.coneUsable ? 1u : 0u;
    p.coneCullEnabled = view.coneCullEnabled ? 1u : 0u;
    p.splitCount = splitCount;
    p.splitsAddress = splitsAddress;
    p.positionsAddress = positionsAddress;
    p.outputsAddress = outputsAddress;
    return p;
}

} // namespace fire_engine

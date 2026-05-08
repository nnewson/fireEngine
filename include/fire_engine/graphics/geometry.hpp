#pragma once

#include <cstdint>
#include <vector>

#include <fire_engine/graphics/draw_command.hpp>
#include <fire_engine/graphics/gpu_handle.hpp>
#include <fire_engine/graphics/vertex.hpp>

namespace fire_engine
{

class Material;
class Resources;

class Geometry
{
public:
    Geometry() = default;
    ~Geometry() = default;

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept = default;
    Geometry& operator=(Geometry&&) noexcept = default;

    void load(Resources& resources);

    [[nodiscard]] bool loaded() const noexcept
    {
        return vertexBuffer_ != NullBuffer;
    }

    [[nodiscard]] const std::vector<Vertex>& vertices() const noexcept
    {
        return vertices_;
    }
    void vertices(std::vector<Vertex> v) noexcept
    {
        vertices_ = std::move(v);
    }

    [[nodiscard]] const std::vector<uint32_t>& indices() const noexcept
    {
        return indices_;
    }
    void indices(std::vector<uint32_t> v) noexcept
    {
        indices_ = std::move(v);
    }
    void indices(std::vector<uint16_t> v) noexcept
    {
        indices_.assign(v.begin(), v.end());
    }

    [[nodiscard]] const Material& material() const noexcept
    {
        return *material_;
    }
    void material(const Material* m) noexcept
    {
        material_ = m;
    }

    [[nodiscard]] BufferHandle vertexBuffer() const noexcept
    {
        return vertexBuffer_;
    }

    [[nodiscard]] BufferHandle indexBuffer() const noexcept
    {
        return indexBuffer_;
    }

    [[nodiscard]] uint32_t indexCount() const noexcept
    {
        return static_cast<uint32_t>(indices_.size());
    }

    [[nodiscard]] DrawIndexType indexType() const noexcept
    {
        return DrawIndexType::UInt32;
    }

    [[nodiscard]] bool castsShadow() const noexcept
    {
        return castsShadow_;
    }
    void castsShadow(bool value) noexcept
    {
        castsShadow_ = value;
    }

    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphPositions() const noexcept
    {
        return morphPositions_;
    }
    void morphPositions(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphPositions_ = std::move(v);
    }

    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphNormals() const noexcept
    {
        return morphNormals_;
    }
    void morphNormals(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphNormals_ = std::move(v);
    }

    // glTF morph TANGENT deltas — Vec3 per vertex per target (no handedness;
    // the base tangent's .w stays unchanged across targets per spec).
    [[nodiscard]] const std::vector<std::vector<Vec3>>& morphTangents() const noexcept
    {
        return morphTangents_;
    }
    void morphTangents(std::vector<std::vector<Vec3>> v) noexcept
    {
        morphTangents_ = std::move(v);
    }

    [[nodiscard]] std::size_t morphTargetCount() const noexcept
    {
        return morphPositions_.size();
    }

private:
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    const Material* material_{nullptr};

    std::vector<std::vector<Vec3>> morphPositions_;
    std::vector<std::vector<Vec3>> morphNormals_;
    std::vector<std::vector<Vec3>> morphTangents_;

    BufferHandle vertexBuffer_{NullBuffer};
    BufferHandle indexBuffer_{NullBuffer};
    bool castsShadow_{true};
};

} // namespace fire_engine

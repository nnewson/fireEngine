#pragma once

#include <fire_engine/graphics/colour3.hpp>
#include <fire_engine/graphics/joints4.hpp>
#include <fire_engine/math/vec2.hpp>
#include <fire_engine/math/vec3.hpp>
#include <fire_engine/math/vec4.hpp>

namespace fire_engine
{

class Vertex
{
public:
    Vertex() = default;
    ~Vertex() = default;

    Vertex(const Vertex&) = default;
    Vertex& operator=(const Vertex&) = default;
    Vertex(Vertex&&) noexcept = default;
    Vertex& operator=(Vertex&&) noexcept = default;

    Vertex(Vec3 position, Colour3 colour, Vec3 normal, Vec2 texCoord) noexcept
        : position_(position),
          colour_(colour),
          normal_(normal),
          texCoord_(texCoord)
    {
    }

    Vertex(Vec3 position, Colour3 colour, Vec3 normal, Vec2 texCoord, Joints4 joints,
           Vec4 weights) noexcept
        : position_(position),
          colour_(colour),
          normal_(normal),
          texCoord_(texCoord),
          joints_(joints),
          weights_(weights)
    {
    }

    Vertex(Vec3 position, Colour3 colour, Vec3 normal, Vec2 texCoord, Joints4 joints, Vec4 weights,
           Vec4 tangent) noexcept
        : position_(position),
          colour_(colour),
          normal_(normal),
          texCoord_(texCoord),
          joints_(joints),
          weights_(weights),
          tangent_(tangent)
    {
    }

    [[nodiscard]] Vec3 position() const noexcept
    {
        return position_;
    }
    void position(Vec3 v) noexcept
    {
        position_ = v;
    }

    [[nodiscard]] Colour3 colour() const noexcept
    {
        return colour_;
    }
    void colour(Colour3 c) noexcept
    {
        colour_ = c;
    }

    [[nodiscard]] Vec3 normal() const noexcept
    {
        return normal_;
    }
    void normal(Vec3 v) noexcept
    {
        normal_ = v;
    }

    [[nodiscard]] Vec2 texCoord() const noexcept
    {
        return texCoord_;
    }
    void texCoord(Vec2 tc) noexcept
    {
        texCoord_ = tc;
    }

    [[nodiscard]] Joints4 joints() const noexcept
    {
        return joints_;
    }
    void joints(Joints4 j) noexcept
    {
        joints_ = j;
    }

    [[nodiscard]] Vec4 weights() const noexcept
    {
        return weights_;
    }
    void weights(Vec4 w) noexcept
    {
        weights_ = w;
    }

    [[nodiscard]] Vec4 tangent() const noexcept
    {
        return tangent_;
    }
    void tangent(Vec4 t) noexcept
    {
        tangent_ = t;
    }

    // Second UV set. glTF allows materials to point individual textures at
    // either TEXCOORD_0 or TEXCOORD_1; absent when the source mesh only
    // declares one set, in which case the loader copies texCoord into here.
    [[nodiscard]] Vec2 texCoord1() const noexcept
    {
        return texCoord1_;
    }
    void texCoord1(Vec2 tc) noexcept
    {
        texCoord1_ = tc;
    }

    [[nodiscard]]
    bool operator==(const Vertex&) const noexcept = default;

    friend class Pipeline;

private:
    Vec3 position_{};
    Colour3 colour_{};
    Vec3 normal_{};
    Vec2 texCoord_{};
    Joints4 joints_{};
    Vec4 weights_{};
    Vec4 tangent_{};
    Vec2 texCoord1_{};
};

} // namespace fire_engine

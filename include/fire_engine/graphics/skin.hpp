#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <fire_engine/math/mat4.hpp>

namespace fire_engine
{

class Node;

class Skin
{
public:
    Skin() = default;
    ~Skin() = default;

    Skin(const Skin&) = delete;
    Skin& operator=(const Skin&) = delete;
    Skin(Skin&&) noexcept = default;
    Skin& operator=(Skin&&) noexcept = default;

    void addJoint(const Node* node, const Mat4& inverseBindMatrix);

    void updateJointMatrices();

    [[nodiscard]] const std::vector<Mat4>& cachedJointMatrices() const noexcept
    {
        return cachedJointMatrices_;
    }

    [[nodiscard]] std::vector<Mat4> computeJointMatrices() const;

    [[nodiscard]] std::size_t jointCount() const noexcept
    {
        return jointNodes_.size();
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return jointNodes_.empty();
    }

    [[nodiscard]] const std::string& name() const noexcept
    {
        return name_;
    }
    void name(std::string n) noexcept
    {
        name_ = std::move(n);
    }

private:
    std::string name_;
    std::vector<const Node*> jointNodes_;
    std::vector<Mat4> inverseBindMatrices_;
    std::vector<Mat4> cachedJointMatrices_;
};

} // namespace fire_engine

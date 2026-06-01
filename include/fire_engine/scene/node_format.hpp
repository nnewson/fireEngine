#pragma once

#include <format>
#include <string>

#include <fire_engine/scene/components.hpp>
#include <fire_engine/scene/node.hpp>
#include <fire_engine/scene/transform.hpp>

namespace fire_engine::detail
{

inline auto formatNode(const fire_engine::Node& node, std::format_context& ctx, int depth)
    -> std::format_context::iterator
{
    auto indent = std::string(static_cast<std::size_t>(depth * 2), ' ');

    ctx.advance_to(std::format_to(ctx.out(), "{}{} [{}] {}", indent, node.name(),
                                  fire_engine::componentName(node.component()), node.transform()));

    for (const auto& child : node.children())
    {
        ctx.advance_to(std::format_to(ctx.out(), "\n"));
        ctx.advance_to(formatNode(*child, ctx, depth + 1));
    }

    return ctx.out();
}

} // namespace fire_engine::detail

template <>
struct std::formatter<fire_engine::Node>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const fire_engine::Node& node, std::format_context& ctx) const
    {
        return fire_engine::detail::formatNode(node, ctx, 0);
    }
};

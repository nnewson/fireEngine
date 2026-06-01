#pragma once

#include <fire_engine/scene/node_format.hpp>
#include <fire_engine/scene/scene_graph.hpp>

template <>
struct std::formatter<fire_engine::SceneGraph>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const fire_engine::SceneGraph& scene, std::format_context& ctx) const
    {
        ctx.advance_to(std::format_to(ctx.out(), "SceneGraph:"));
        for (const auto& node : scene.nodes())
        {
            ctx.advance_to(std::format_to(ctx.out(), "\n"));
            ctx.advance_to(fire_engine::detail::formatNode(*node, ctx, 1));
        }
        return ctx.out();
    }
};

#include "fire_engine/graphics/shadow_view_disposition.hpp"

namespace fire_engine
{

std::string_view toString(ShadowViewDisposition disposition) noexcept
{
    switch (disposition)
    {
    case ShadowViewDisposition::Invalid:
        return "invalid";
    case ShadowViewDisposition::Reused:
        return "reused";
    case ShadowViewDisposition::Recorded:
        return "recorded";
    }
    return "unknown";
}

} // namespace fire_engine

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fire_engine::log
{

enum class Level
{
    Debug = 0,
    Info,
    Warn,
    Error,
    Off
};

struct Category
{
    std::string_view name;
};

namespace category
{
inline constexpr Category app{"app"};
inline constexpr Category general{"general"};
inline constexpr Category gltf{"gltf"};
inline constexpr Category physics{"physics"};
inline constexpr Category ragdoll{"ragdoll"};
inline constexpr Category render{"render"};
} // namespace category

namespace detail
{

struct Rule
{
    std::string category;
    Level level{Level::Warn};
};

struct Config
{
    Level global{Level::Warn};
    std::vector<Rule> rules;
};

[[nodiscard]]
inline std::string trim(std::string_view value)
{
    const auto begin =
        std::ranges::find_if_not(value, [](unsigned char c) { return std::isspace(c); });
    const auto end = std::ranges::find_if_not(value.rbegin(), value.rend(),
                                              [](unsigned char c) { return std::isspace(c); })
                         .base();
    if (begin >= end)
    {
        return {};
    }
    return {begin, end};
}

[[nodiscard]]
inline std::string lower(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value)
    {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

[[nodiscard]]
inline Level parseLevel(std::string_view value, Level fallback) noexcept
{
    const std::string v = lower(trim(value));
    if (v == "debug")
    {
        return Level::Debug;
    }
    if (v == "info")
    {
        return Level::Info;
    }
    if (v == "warn" || v == "warning")
    {
        return Level::Warn;
    }
    if (v == "error")
    {
        return Level::Error;
    }
    if (v == "off" || v == "none" || v == "quiet")
    {
        return Level::Off;
    }
    return fallback;
}

inline void parseToken(Config& config, std::string_view token)
{
    const std::string cleaned = trim(token);
    if (cleaned.empty())
    {
        return;
    }

    const std::size_t colon = cleaned.find(':');
    if (colon == std::string::npos)
    {
        config.global = parseLevel(cleaned, config.global);
        return;
    }

    const std::string categoryName = lower(trim(std::string_view{cleaned}.substr(0, colon)));
    if (categoryName.empty())
    {
        return;
    }
    const Level level = parseLevel(std::string_view{cleaned}.substr(colon + 1), config.global);
    if (categoryName == "*" || categoryName == "all" || categoryName == "global")
    {
        config.global = level;
        return;
    }
    config.rules.push_back(Rule{categoryName, level});
}

[[nodiscard]]
inline Config parseConfig()
{
    Config config;
    const char* env = std::getenv("FE_LOG");
    if (env == nullptr)
    {
        return config;
    }

    std::string_view rest{env};
    while (!rest.empty())
    {
        const std::size_t end = rest.find_first_of(",;");
        parseToken(config, rest.substr(0, end));
        if (end == std::string_view::npos)
        {
            break;
        }
        rest.remove_prefix(end + 1);
    }
    return config;
}

[[nodiscard]]
inline const Config& config()
{
    static const Config value = parseConfig();
    return value;
}

[[nodiscard]]
inline Level levelFor(Category categoryValue)
{
    const Config& cfg = config();
    const std::string key = lower(categoryValue.name);
    for (auto it = cfg.rules.rbegin(); it != cfg.rules.rend(); ++it)
    {
        if (it->category == key)
        {
            return it->level;
        }
    }
    return cfg.global;
}

[[nodiscard]]
inline bool enabled(Level messageLevel, Category categoryValue)
{
    const Level configured = levelFor(categoryValue);
    return configured != Level::Off &&
           static_cast<int>(messageLevel) >= static_cast<int>(configured);
}

[[nodiscard]]
inline std::string_view levelName(Level level) noexcept
{
    switch (level)
    {
    case Level::Debug:
        return "debug";
    case Level::Info:
        return "info";
    case Level::Warn:
        return "warn";
    case Level::Error:
        return "error";
    case Level::Off:
    default:
        return "off";
    }
}

inline std::mutex& outputMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline void write(Level level, Category categoryValue, std::string_view message)
{
    std::lock_guard lock{outputMutex()};
    std::fputs("[", stderr);
    std::fwrite(levelName(level).data(), 1, levelName(level).size(), stderr);
    std::fputs("][", stderr);
    std::fwrite(categoryValue.name.data(), 1, categoryValue.name.size(), stderr);
    std::fputs("] ", stderr);
    std::fwrite(message.data(), 1, message.size(), stderr);
    if (message.empty() || message.back() != '\n')
    {
        std::fputc('\n', stderr);
    }
}

template <typename... Args>
inline void log(Level level, Category categoryValue, std::format_string<Args...> fmt,
                Args&&... args)
{
    if (!enabled(level, categoryValue))
    {
        return;
    }
    write(level, categoryValue, std::format(fmt, std::forward<Args>(args)...));
}

} // namespace detail

template <typename... Args>
inline void debug(Category categoryValue, std::format_string<Args...> fmt, Args&&... args)
{
    detail::log(Level::Debug, categoryValue, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void info(Category categoryValue, std::format_string<Args...> fmt, Args&&... args)
{
    detail::log(Level::Info, categoryValue, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void warn(Category categoryValue, std::format_string<Args...> fmt, Args&&... args)
{
    detail::log(Level::Warn, categoryValue, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(Category categoryValue, std::format_string<Args...> fmt, Args&&... args)
{
    detail::log(Level::Error, categoryValue, fmt, std::forward<Args>(args)...);
}

[[nodiscard]]
inline bool enabled(Level level, Category categoryValue)
{
    return detail::enabled(level, categoryValue);
}

template <typename... Args>
inline void debug(std::format_string<Args...> fmt, Args&&... args)
{
    debug(category::general, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void info(std::format_string<Args...> fmt, Args&&... args)
{
    info(category::general, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void warn(std::format_string<Args...> fmt, Args&&... args)
{
    warn(category::general, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(std::format_string<Args...> fmt, Args&&... args)
{
    error(category::general, fmt, std::forward<Args>(args)...);
}

} // namespace fire_engine::log

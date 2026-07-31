#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <pwd.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"

namespace sdk_service_trace {

struct TraceConfig {
    bool enable{false};
    bool console{false};
    bool file{false};
    bool sdk_service_trace{false};
    std::string log_dir;
};

inline std::mutex g_file_mutex;
inline std::once_flag g_dir_once;
inline std::once_flag g_session_once;
inline std::once_flag g_config_once;
inline std::shared_ptr<std::ofstream> g_stream;
inline bool g_open_failed{false};
inline std::filesystem::path g_session_dir;
inline std::string g_session_stamp;
inline TraceConfig g_config;

inline std::filesystem::path GetCurrentUserHomeDir() {
    const char* home_env = std::getenv("HOME");
    if (home_env != nullptr && home_env[0] != '\0') {
        return std::filesystem::path(home_env);
    }

    struct passwd* pw = getpwuid(getuid());
    if (pw != nullptr && pw->pw_dir != nullptr && pw->pw_dir[0] != '\0') {
        return std::filesystem::path(pw->pw_dir);
    }

    return std::filesystem::temp_directory_path();
}

inline std::string SanitizeName(std::string value) {
    for (char& ch : value) {
        const bool is_valid =
            (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
            ch == '_' || ch == '-';
        if (!is_valid) {
            ch = '_';
        }
    }
    if (value.empty()) {
        value = "process";
    }
    return value;
}

inline std::string Trim(std::string value) {
    const auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
    return value;
}

inline std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline std::string StripQuotes(std::string value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

inline bool ParseBool(const std::string& value, bool default_value) {
    const std::string lowered = ToLower(Trim(value));
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return default_value;
}

inline std::filesystem::path GetConfigPath() {
    try {
        return std::filesystem::path(
                   ament_index_cpp::get_package_share_directory("lite3_sdk_service")) /
               "config/trace.yaml";
    } catch (...) {
        return {};
    }
}

inline void LoadConfig() {
    std::call_once(g_config_once, []() {
        g_config = TraceConfig{};

        const std::filesystem::path config_path = GetConfigPath();
        if (config_path.empty()) {
            return;
        }

        std::ifstream input(config_path);
        if (!input.is_open()) {
            return;
        }

        std::string line;
        while (std::getline(input, line)) {
            const std::size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            line = Trim(line);
            if (line.empty()) {
                continue;
            }

            const std::size_t sep = line.find(':');
            if (sep == std::string::npos) {
                continue;
            }

            const std::string key = ToLower(Trim(line.substr(0, sep)));
            const std::string value = StripQuotes(Trim(line.substr(sep + 1)));

            if (key == "enable") {
                g_config.enable = ParseBool(value, g_config.enable);
            } else if (key == "console") {
                g_config.console = ParseBool(value, g_config.console);
            } else if (key == "file") {
                g_config.file = ParseBool(value, g_config.file);
            } else if (key == "sdk_service_trace") {
                g_config.sdk_service_trace = ParseBool(value, g_config.sdk_service_trace);
            } else if (key == "log_dir") {
                g_config.log_dir = value;
            }
        }
    });
}

inline bool IsTraceEnabled() {
    LoadConfig();
    return g_config.enable;
}

inline bool IsConsoleTraceEnabled() {
    LoadConfig();
    return g_config.enable && g_config.console;
}

inline bool IsFileTraceEnabled() {
    LoadConfig();
    return g_config.enable && g_config.file;
}

inline bool IsSdkServiceTraceEnabled() {
    LoadConfig();
    return g_config.enable && g_config.sdk_service_trace;
}

inline std::string FormatSessionStamp() {
    using Clock = std::chrono::system_clock;
    const auto now = Clock::now();
    const auto secs = Clock::to_time_t(now);

    std::tm local_tm{};
    localtime_r(&secs, &local_tm);

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

inline std::string GetProcessTag() {
    const char* env_tag = std::getenv("TOPIC_TRACE_SESSION_TAG");
    if (env_tag != nullptr && env_tag[0] != '\0') {
        return SanitizeName(env_tag);
    }

    std::error_code ec;
    const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe_path.empty()) {
        return SanitizeName(exe_path.filename().string());
    }

    return "process";
}

inline std::filesystem::path GetLogDir() {
    LoadConfig();
    if (!g_config.log_dir.empty()) {
        return std::filesystem::path(g_config.log_dir);
    }
    return GetCurrentUserHomeDir() / "ros2 debug log";
}

inline void EnsureLogDirExists() {
    std::call_once(g_dir_once, []() {
        std::error_code ec;
        std::filesystem::create_directories(GetLogDir(), ec);
        if (ec) {
            std::cerr << "[sdk_service_trace] failed to create log directory: "
                      << GetLogDir().string() << " error=" << ec.message() << std::endl;
        }
    });
}

inline void EnsureSessionDirExists() {
    std::call_once(g_session_once, []() {
        EnsureLogDirExists();

        g_session_stamp = FormatSessionStamp();
        const std::string process_tag = GetProcessTag();
        g_session_dir =
            GetLogDir() / (g_session_stamp + "_" + process_tag + "_pid" + std::to_string(getpid()));

        std::error_code ec;
        std::filesystem::create_directories(g_session_dir, ec);
        if (ec) {
            std::cerr << "[sdk_service_trace] failed to create session directory: "
                      << g_session_dir.string() << " error=" << ec.message() << std::endl;
            g_session_dir.clear();
        }
    });
}

inline std::string FormatWallTime() {
    using Clock = std::chrono::system_clock;
    const auto now = Clock::now();
    const auto secs = Clock::to_time_t(now);
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() %
        1000000;

    std::tm local_tm{};
    localtime_r(&secs, &local_tm);

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << '.'
        << std::setw(6) << std::setfill('0') << micros;
    return oss.str();
}

inline double NowSteadySec() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

inline std::shared_ptr<std::ofstream> GetStreamLocked() {
    if (!IsFileTraceEnabled()) {
        return nullptr;
    }
    EnsureSessionDirExists();
    if (g_stream) {
        return g_stream;
    }

    const std::filesystem::path log_root = g_session_dir.empty() ? GetLogDir() : g_session_dir;
    const std::string stamp = g_session_stamp.empty() ? FormatSessionStamp() : g_session_stamp;
    const auto log_path = log_root / ("SDK_SERVICE_EVENT_" + stamp + ".log");
    g_stream = std::make_shared<std::ofstream>(log_path, std::ios::app);
    if (!g_stream->is_open()) {
        if (!g_open_failed) {
            std::cerr << "[sdk_service_trace] failed to open log file: " << log_path.string()
                      << std::endl;
            g_open_failed = true;
        }
        g_stream.reset();
        return nullptr;
    }

    return g_stream;
}

inline void LogEvent(
    const rclcpp::Logger& logger,
    const std::string& tag,
    const std::string& message) {
    if (!IsSdkServiceTraceEnabled()) {
        return;
    }
    const double steady_now_sec = NowSteadySec();
    const std::string full_msg = tag + " " + message;

    if (IsConsoleTraceEnabled()) {
        RCLCPP_WARN(logger, "%s", full_msg.c_str());
    }

    if (!IsFileTraceEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_file_mutex);
    auto stream = GetStreamLocked();
    if (!stream) {
        return;
    }

    (*stream) << FormatWallTime()
              << " steady=" << std::fixed << std::setprecision(6) << steady_now_sec
              << " " << full_msg
              << '\n';
    stream->flush();
}

}  // namespace sdk_service_trace

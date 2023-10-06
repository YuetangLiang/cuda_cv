/******************************************************************************
 *****************************************************************************/
#pragma once
#include <iostream>

namespace lad {
namespace lcv {
namespace cuda {

#define TERM_NORMAL "\033[0m"
#define TERM_RED "\033[0;31m"
#define TERM_YELLOW "\033[0;33m"
#define TERM_GREEN "\033[0;32m"
#define TERM_MAGENTA "\033[1;35m"

enum class LogLevel : uint8_t {
  kERROR = 0,
  kWARNING = 1,
  kINFO = 2,
  kDEBUG = 3,
};

inline void log(LogLevel log_level, LogLevel reportable_severity, std::string msg) {
  // suppress messages with severity enum value greater than the reportable
  if (log_level > reportable_severity) {
    return;
  }

  switch (log_level) {
    case LogLevel::kERROR:
      std::cerr << TERM_RED;
      break;
    case LogLevel::kWARNING:
      std::cerr << TERM_YELLOW;
      break;
    case LogLevel::kINFO:
      std::cerr << TERM_GREEN;
      break;
    case LogLevel::kDEBUG:
      std::cerr << TERM_MAGENTA;
      break;
    default:
      break;
  }

  switch (log_level) {
    case LogLevel::kERROR:
      std::cerr << "ERROR: ";
      break;
    case LogLevel::kWARNING:
      std::cerr << "WARNING: ";
      break;
    case LogLevel::kINFO:
      std::cerr << "INFO: ";
      break;
    case LogLevel::kDEBUG:
      std::cerr << "DEBUG: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
      break;
  }

  std::cerr << TERM_NORMAL;
  std::cerr << msg << std::endl;
}

#ifdef CUDA_DEBUG_LOG
#define SEVERITY LogLevel::kDEBUG
#else
#define SEVERITY LogLevel::kINFO
#endif

#define GET_MACRO(NAME, ...) NAME

#define CUDA_LOG(l, sev, msg) \
  do {                        \
    std::stringstream ss{};   \
    ss << msg;                \
    log(l, sev, ss.str());    \
  } while (0)

#define LOG_DEBUG_GLOBAL(s) CUDA_LOG(LogLevel::kDEBUG, SEVERITY, s)
#define LOG_INFO_GLOBAL(s) CUDA_LOG(LogLevel::kINFO, SEVERITY, s)
#define LOG_WARNING_GLOBAL(s) CUDA_LOG(LogLevel::kWARNING, SEVERITY, s)
#define LOG_ERROR_GLOBAL(s) CUDA_LOG(LogLevel::kERROR, SEVERITY, s)

#define LOG_DEBUG(...) GET_MACRO(LOG_DEBUG_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_INFO(...) GET_MACRO(LOG_INFO_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_WARNING(...) GET_MACRO(LOG_WARNING_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_ERROR(...) GET_MACRO(LOG_ERROR_GLOBAL, __VA_ARGS__)(__VA_ARGS__)

}  // namespace cuda
}  // namespace lcv
}  // namespace lad

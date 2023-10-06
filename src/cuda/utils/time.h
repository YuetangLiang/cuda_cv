/******************************************************************************
 *****************************************************************************/
#pragma once
#include <chrono>
namespace lad {
namespace lcv {
namespace cuda {

inline uint64_t system_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
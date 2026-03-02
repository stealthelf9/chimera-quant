#pragma once
#include <cstdint>

namespace chimera {

// Core OHLCV struct, aligned to 32 bytes for memory efficiency and vectorized
// operations.
struct alignas(32) OHLCV {
  uint64_t timestamp; // Unix timestamp in nanoseconds
  double open;
  double high;
  double low;
  double close;
  uint64_t volume;
  uint32_t instrument_id;
};

} // namespace chimera

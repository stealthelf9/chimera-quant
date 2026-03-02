#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace chimera {

class DbnParser {
public:
  // Parses a .dbn.zst file and appends OHLCV records to the buffer array
  static void parse_zstd_dbn(const std::string &filepath,
                             std::vector<OHLCV> &buffer);
};

} // namespace chimera

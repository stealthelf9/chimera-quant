#include "dbn_parser.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <zstd.h>

namespace chimera {

#pragma pack(push, 1)
struct DbnRecordHeader {
  uint8_t length;
  uint8_t rtype;
  uint16_t publisher_id;
  uint32_t instrument_id;
  uint64_t ts_event;
};

struct DbnOhlcvMsg {
  DbnRecordHeader hd;
  int64_t open;
  int64_t high;
  int64_t low;
  int64_t close;
  uint64_t volume;
};
#pragma pack(pop)

void DbnParser::parse_zstd_dbn(const std::string &filepath,
                               std::vector<OHLCV> &buffer) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filepath);
  }

  ZSTD_DStream *dstream = ZSTD_createDStream();
  if (dstream == nullptr) {
    throw std::runtime_error("Failed to create ZSTD DStream");
  }
  size_t const initResult = ZSTD_initDStream(dstream);
  if (ZSTD_isError(initResult)) {
    ZSTD_freeDStream(dstream);
    throw std::runtime_error("ZSTD_initDStream failed");
  }

  size_t const buffInSize = ZSTD_DStreamInSize();
  std::vector<char> buffIn(buffInSize);
  size_t const buffOutSize = ZSTD_DStreamOutSize();
  std::vector<char> buffOut(buffOutSize);

  std::vector<char> decompressed_data;
  // Pre-allocate to prevent excessive re-allocations
  decompressed_data.reserve(256 * 1024 * 1024); // 256MB chunks if needed

  while (file.read(buffIn.data(), buffIn.size()) || file.gcount() > 0) {
    size_t const toRead = file.gcount();
    ZSTD_inBuffer input = {buffIn.data(), toRead, 0};
    while (input.pos < input.size) {
      ZSTD_outBuffer output = {buffOut.data(), buffOut.size(), 0};
      size_t const ret = ZSTD_decompressStream(dstream, &output, &input);
      if (ZSTD_isError(ret)) {
        ZSTD_freeDStream(dstream);
        throw std::runtime_error("ZSTD decompression failed");
      }
      decompressed_data.insert(decompressed_data.end(), buffOut.begin(),
                               buffOut.begin() + output.pos);
    }
  }

  ZSTD_freeDStream(dstream);

  // Parse DBN
  if (decompressed_data.size() < 8) {
    throw std::runtime_error("File too small for DBN header");
  }

  if (std::strncmp(decompressed_data.data(), "DBN", 3) != 0) {
    throw std::runtime_error("Not a DBN file");
  }

  uint32_t metadata_length =
      *reinterpret_cast<const uint32_t *>(decompressed_data.data() + 4);
  size_t offset = 8 + metadata_length;

  while (offset + sizeof(DbnRecordHeader) <= decompressed_data.size()) {
    const DbnRecordHeader *header = reinterpret_cast<const DbnRecordHeader *>(
        decompressed_data.data() + offset);
    size_t record_bytes = header->length * 4;

    if (offset + record_bytes > decompressed_data.size()) {
      break;
    }

    // Filter based on typical OHLCV record size (56 bytes) and ensure length
    // matches
    if (record_bytes == sizeof(DbnOhlcvMsg)) {
      const DbnOhlcvMsg *msg = reinterpret_cast<const DbnOhlcvMsg *>(
          decompressed_data.data() + offset);
      OHLCV ohlcv;
      // DBN timestamps are usually nanoseconds since epoch
      ohlcv.timestamp = msg->hd.ts_event;
      // DBN prices are fixed-point with a 1e9 multiplier.
      ohlcv.open = static_cast<double>(msg->open) / 1e9;
      ohlcv.high = static_cast<double>(msg->high) / 1e9;
      ohlcv.low = static_cast<double>(msg->low) / 1e9;
      ohlcv.close = static_cast<double>(msg->close) / 1e9;
      ohlcv.volume = msg->volume;

      // Basic sanity check to avoid weird data
      if (ohlcv.close > 0.0 && ohlcv.high >= ohlcv.low) {
        buffer.push_back(ohlcv);
      }
    }

    offset += record_bytes;
  }
}

} // namespace chimera

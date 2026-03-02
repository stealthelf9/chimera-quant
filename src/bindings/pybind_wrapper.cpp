#include "../core/backtest.h"
#include "../core/dbn_parser.h"
#include "../core/types.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace chimera;

// Buffer to hold market data
class MarketDataBuffer {
public:
  MarketDataBuffer(size_t reserve_size = 100000) { data.reserve(reserve_size); }

  void append(const OHLCV &tick) { data.push_back(tick); }

  // Zero-copy NumPy view using pybind11 buffer protocol
  py::array_t<OHLCV> get_buffer_view() {
    // We create a NumPy array pointing to the std::vector data without copying.
    // We tie the lifetime of the array to the MarketDataBuffer instance
    // by passing py::cast(this) as the base handle for reference counting.
    return py::array_t<OHLCV>(data.size(),   // shape
                              data.data(),   // mutable data pointer
                              py::cast(this) // base object
    );
  }

  // Load data from DBN ZSTD file
  void load_dbn(const std::string &filepath) {
    DbnParser::parse_zstd_dbn(filepath, data);
  }

  // Dynamically resample 1-minute data to larger timeframes (e.g., 5, 15, 60,
  // 1440)
  std::shared_ptr<MarketDataBuffer> resample(int interval_minutes) const {
    auto resampled =
        std::make_shared<MarketDataBuffer>(data.size() / interval_minutes + 1);
    if (data.empty() || interval_minutes <= 1) {
      resampled->data = data;
      return resampled;
    }

    uint64_t interval_ns =
        static_cast<uint64_t>(interval_minutes) * 60 * 1000000000ULL;

    // Using the first tick's timestamp to align intervals.
    // Usually, you might want to align to trading day open or top of the hour.
    // For simplicity, we just chunk by the specified interval.

    OHLCV current_candle = data[0];
    uint64_t current_interval_end =
        ((current_candle.timestamp / interval_ns) + 1) * interval_ns;

    for (size_t i = 1; i < data.size(); ++i) {
      const auto &tick = data[i];
      if (tick.timestamp < current_interval_end) {
        // Update candle
        current_candle.high = std::max(current_candle.high, tick.high);
        current_candle.low = std::min(current_candle.low, tick.low);
        current_candle.close = tick.close;
        current_candle.volume += tick.volume;
      } else {
        // Close candle and start new
        resampled->append(current_candle);
        current_candle = tick;
        current_interval_end =
            ((current_candle.timestamp / interval_ns) + 1) * interval_ns;
      }
    }
    resampled->append(current_candle); // Append the final candle
    return resampled;
  }

  BacktestStats run_backtest(double initial_capital, double order_size,
                             bool size_is_percentage, double commission,
                             bool commission_is_percentage,
                             double slippage_penalty, uint64_t start_timestamp,
                             uint64_t end_timestamp,
                             const std::vector<int> &signals) {
    BacktestSimulator sim(data);
    return sim.run(initial_capital, order_size, size_is_percentage, commission,
                   commission_is_percentage, slippage_penalty, start_timestamp,
                   end_timestamp, signals);
  }

  std::vector<OHLCV> data;
};

PYBIND11_MODULE(chimera_core, m) {
  m.doc() = "Chimera Quant Core Engine";

  // Register custom NumPy dtype for OHLCV
  PYBIND11_NUMPY_DTYPE(OHLCV, timestamp, open, high, low, close, volume);

  py::class_<OHLCV>(m, "OHLCV")
      .def(py::init<>())
      .def_readwrite("timestamp", &OHLCV::timestamp)
      .def_readwrite("open", &OHLCV::open)
      .def_readwrite("high", &OHLCV::high)
      .def_readwrite("low", &OHLCV::low)
      .def_readwrite("close", &OHLCV::close)
      .def_readwrite("volume", &OHLCV::volume);

  py::class_<BacktestStats>(m, "BacktestStats")
      .def_readonly("net_profit_usd", &BacktestStats::net_profit_usd)
      .def_readonly("net_profit_pct", &BacktestStats::net_profit_pct)
      .def_readonly("win_rate", &BacktestStats::win_rate)
      .def_readonly("max_drawdown", &BacktestStats::max_drawdown)
      .def_readonly("sharpe_ratio", &BacktestStats::sharpe_ratio);

  py::class_<MarketDataBuffer, std::shared_ptr<MarketDataBuffer>>(
      m, "MarketDataBuffer")
      .def(py::init<size_t>(), py::arg("reserve_size") = 100000)
      .def("append", &MarketDataBuffer::append)
      .def("load_dbn", &MarketDataBuffer::load_dbn, py::arg("filepath"))
      .def("resample", &MarketDataBuffer::resample, py::arg("interval_minutes"))
      .def("run_backtest", &MarketDataBuffer::run_backtest,
           py::arg("initial_capital"), py::arg("order_size"),
           py::arg("size_is_percentage"), py::arg("commission"),
           py::arg("commission_is_percentage"), py::arg("slippage_penalty"),
           py::arg("start_timestamp"), py::arg("end_timestamp"),
           py::arg("signals"))
      .def("get_buffer_view", &MarketDataBuffer::get_buffer_view,
           py::return_value_policy::reference_internal);
}

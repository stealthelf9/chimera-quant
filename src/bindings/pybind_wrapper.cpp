#include "../core/dbn_parser.h"
#include "../core/types.h"
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

  py::class_<MarketDataBuffer>(m, "MarketDataBuffer")
      .def(py::init<size_t>(), py::arg("reserve_size") = 100000)
      .def("append", &MarketDataBuffer::append)
      .def("load_dbn", &MarketDataBuffer::load_dbn, py::arg("filepath"))
      .def("get_buffer_view", &MarketDataBuffer::get_buffer_view,
           py::return_value_policy::reference_internal);
}

cmake_minimum_required(VERSION 3.14)
project(ChimeraQuant VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# High-performance compiler flags (optimized for AMD Ryzen/Zen-architecture)
set(CMAKE_CXX_FLAGS "-O3 -march=native -flto -fPIC -Wall -Wextra")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11 if not installed globally
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find libzstd
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Pybind11 module target
pybind11_add_module(chimera_core src/bindings/pybind_wrapper.cpp src/core/dbn_parser.cpp src/core/backtest.cpp)

target_include_directories(chimera_core PRIVATE src/core ${ZSTD_INCLUDE_DIRS})
target_link_libraries(chimera_core PRIVATE ${ZSTD_LIBRARIES})


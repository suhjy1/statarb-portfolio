cmake_minimum_required(VERSION 3.25)
project(statarb_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# -O2 for release builds instead of CMake's default -O3
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

option(STATARB_SANITIZE "Build with -fsanitize=address,undefined" OFF)
option(STATARB_BUILD_PYTHON "Build the pybind11 module" ON)
option(STATARB_BUILD_TESTS "Build the GoogleTest suite" ON)

add_compile_options(-Wall -Wextra)
if(STATARB_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# core library with the engine and estimators, no Python dependency. Add new sources here
add_library(statarb_core STATIC
  src/backtest.cpp
  src/ou.cpp
)
target_include_directories(statarb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(STATARB_BUILD_PYTHON AND NOT STATARB_SANITIZE)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(statarb_cpp bindings/module.cpp)
  target_link_libraries(statarb_cpp PRIVATE statarb_core)
  # pytest picks the module up from here, see pythonpath in pyproject.toml
  set_target_properties(statarb_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
endif()

if(STATARB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

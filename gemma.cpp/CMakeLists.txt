# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Baseline minimum lowered previously; require at least 3.5 but declare
# compatibility up to 3.27 for modern policies when available.
cmake_minimum_required(VERSION 3.5...3.27)

# Handle policy compatibility for dependencies
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# Set policy defaults for FetchContent dependencies
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # option() honors normal variables
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)  # Include file check macros honor CMAKE_REQUIRED_LIBRARIES

include(FetchContent)

# vcpkg integration handled at project root (before first project()).
# We still allow users to prefer system/vcpkg packages by default.
option(GEMMA_PREFER_SYSTEM_DEPS "Use system/vcpkg packages before FetchContent" ON)
if(GEMMA_USING_VCPKG)
  message(STATUS "[vcpkg] Root toolchain active inside gemma.cpp subproject")
endif()

project(gemma)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include optimization module
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(GemmaOptimizations)

# Setup global build optimizations
setup_gemma_build_optimizations()
setup_build_profiles()

# Build optimization options
option(GEMMA_ENABLE_UNITY_BUILDS "Enable unity builds for faster compilation" OFF)
option(GEMMA_ENABLE_LTO "Enable Link Time Optimization" ON)
option(GEMMA_ENABLE_PCH "Enable Precompiled Headers" ON)
option(GEMMA_REQUIRE_HWY_CONTRIB "Fail configuration if hwy_contrib (Highway contrib) is not available" OFF)

if(GEMMA_PREFER_SYSTEM_DEPS)
  # vcpkg exposes 'highway' as the target 'hwy'. Some ports also create hwy_contrib.
  find_package(highway QUIET CONFIG)
endif()
if(NOT TARGET hwy)
  message(STATUS "highway not provided by system/vcpkg; fetching via FetchContent")
  FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 1d16731233de45a365b43867f27d0a5f73925300 EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(highway)
endif()
# Collect Highway related targets (hwy is required, hwy_contrib optional).
set(GEMMA_HWY_LIBS hwy)
if(TARGET hwy_contrib)
  list(APPEND GEMMA_HWY_LIBS hwy_contrib)
else()
  message(STATUS "highway contrib target (hwy_contrib) not available; continuing without it")
  if(GEMMA_REQUIRE_HWY_CONTRIB)
    message(FATAL_ERROR "GEMMA_REQUIRE_HWY_CONTRIB=ON but hwy_contrib target not found. Install full highway package or disable this option.")
  endif()
endif()

## Note: absl needs to be installed by sentencepiece. This will only happen if
## cmake is invoked with -DSPM_ENABLE_SHARED=OFF and -DSPM_ABSL_PROVIDER=module

# Configure sentencepiece build options before fetching
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Disable shared libraries for sentencepiece")
set(SPM_ABSL_PROVIDER "module" CACHE STRING "Use module provider for absl")
set(SPM_BUILD_TEST OFF CACHE BOOL "Disable sentencepiece tests")
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Disable tcmalloc")

if(GEMMA_PREFER_SYSTEM_DEPS)
  find_package(sentencepiece QUIET CONFIG)
endif()
if(NOT TARGET sentencepiece-static AND NOT TARGET sentencepiece)
  message(STATUS "sentencepiece not found via system/vcpkg; using FetchContent")
  FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(sentencepiece)
  if(NOT sentencepiece_POPULATED)
    message(STATUS "Fetching sentencepiece with CMake compatibility fixes...")
    FetchContent_Populate(sentencepiece)
    set(SENTENCEPIECE_CMAKE_FIX "${sentencepiece_SOURCE_DIR}/CMakeLists.txt.fixed")
    file(READ "${sentencepiece_SOURCE_DIR}/CMakeLists.txt" SENTENCEPIECE_CMAKE_CONTENT)
    string(REPLACE "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)"
                   "cmake_minimum_required(VERSION 3.5 FATAL_ERROR)"
                   SENTENCEPIECE_CMAKE_CONTENT "${SENTENCEPIECE_CMAKE_CONTENT}")
    file(WRITE "${SENTENCEPIECE_CMAKE_FIX}" "${SENTENCEPIECE_CMAKE_CONTENT}")
    file(RENAME "${SENTENCEPIECE_CMAKE_FIX}" "${sentencepiece_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
else()
  message(STATUS "sentencepiece found via system/vcpkg: using existing target(s)")
endif()

if(GEMMA_PREFER_SYSTEM_DEPS)
  find_package(nlohmann_json QUIET CONFIG)
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
  message(STATUS "nlohmann_json not found via system/vcpkg; fetching")
  FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(json)
else()
  message(STATUS "nlohmann_json found via system/vcpkg")
endif()

# Configure Google Benchmark (only if not already available)
if(GEMMA_PREFER_SYSTEM_DEPS)
  find_package(benchmark QUIET CONFIG)
endif()
if(NOT TARGET benchmark::benchmark)
  message(STATUS "google benchmark not found via system/vcpkg; fetching")
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
  set(BENCHMARK_ENABLE_INSTALL OFF)

  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.2
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(benchmark)
else()
  message(STATUS "google benchmark found via system/vcpkg")
endif()

# Base source files
set(SOURCES
  compression/compress-inl.h
  compression/compress.cc
  compression/compress.h
  compression/nuq-inl.h
  compression/sfp-inl.h
  compression/types.h
  compression/test_util-inl.h
  evals/benchmark_helper.cc
  evals/benchmark_helper.h
  evals/cross_entropy.cc
  evals/cross_entropy.h
  gemma/activations.h
  gemma/attention.cc
  gemma/attention.h
  gemma/configs.cc
  gemma/configs.h
  gemma/context.cc
  gemma/context.h
  gemma/gemma_args.h
  gemma/gemma-inl.h
  gemma/gemma.cc
  gemma/gemma.h
  gemma/griffin.h
  gemma/kv_cache.cc
  gemma/kv_cache.h
  gemma/model_store.cc
  gemma/model_store.h
  gemma/tensor_info.cc
  gemma/tensor_info.h
  gemma/tokenizer.cc
  gemma/tokenizer.h
  gemma/vit.cc
  gemma/vit.h
  gemma/weights.cc
  gemma/weights.h
  io/blob_store.cc
  io/blob_store.h
  io/fields.cc
  io/fields.h
  io/io_win.cc
  io/io.cc
  io/io.h
  ops/dot-inl.h
  ops/dry_penalty.cc
  ops/dry_penalty.h
  ops/matmul_static_bf16.cc
  ops/matmul_static_f32.cc
  ops/matmul_static_nuq.cc
  ops/matmul_static_sfp.cc
  ops/matmul-inl.h
  ops/matmul.cc
  ops/matmul.h
  ops/matvec-inl.h
  ops/ops-inl.h
  ops/ops.h
  ops/sum-inl.h
  paligemma/image.cc
  paligemma/image.h
  util/allocator.cc
  util/allocator.h
  util/basics.h
  util/mat.cc
  util/mat.h
  util/test_util.h
  util/threading_context.cc
  util/threading_context.h
  util/threading.cc
  util/threading.h
  util/topology.cc
  util/topology.h
)

# Optionally disable the full Griffin (recurrent) implementation to speed up
# builds or work around platform issues. When disabled, a stub providing the
# GriffinRecurrent symbol is used so other translation units still link.
option(GEMMA_DISABLE_GRIFFIN "Build without full Griffin recurrent layer (use stub)" OFF)
option(GEMMA_EXCLUDE_ARCHIVE "Exclude .archive directory from packaging/install" ON)
option(GEMMA_ENABLE_MCP "Build optional MCP server (requires mcp directory)" OFF)
option(GEMMA_ENABLE_CUDA "Enable CUDA backend if available" ON)
option(GEMMA_ENABLE_OPENCL "Enable OpenCL backend if available" ON)
option(GEMMA_ENABLE_SYCL "Enable SYCL backend if available" ON)
option(GEMMA_ENABLE_VULKAN "Enable Vulkan backend if available" OFF)
option(GEMMA_ENABLE_AUTO_BACKENDS "Auto-detect and enable available GPU backends" ON)

if(GEMMA_DISABLE_GRIFFIN)
  message(STATUS "GEMMA_DISABLE_GRIFFIN=ON: using gemma/griffin_stub.cc (no recurrent layer logic)")
  list(APPEND SOURCES gemma/griffin_stub.cc)
else()
  list(APPEND SOURCES gemma/griffin.cc)
endif()

# Add C API sources only when building DLL
if(BUILD_GEMMA_DLL)
  list(APPEND SOURCES
    gemma/bindings/context.h
    gemma/bindings/context.cc
    gemma/bindings/c_api.h
    gemma/bindings/c_api.cc
  )
  message(STATUS "Including C API files for DLL build")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

FetchContent_GetProperties(sentencepiece)

## Library Target

add_library(libgemma ${SOURCES})
set_property(TARGET libgemma PROPERTY CXX_STANDARD 20)
set_target_properties(libgemma PROPERTIES PREFIX "")
set_property(TARGET libgemma PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(libgemma PUBLIC ./ ./gemma)
if(TARGET sentencepiece-static)
  set(GEMMA_SENTENCEPIECE_LIB sentencepiece-static)
elseif(TARGET sentencepiece)
  set(GEMMA_SENTENCEPIECE_LIB sentencepiece)
else()
  message(FATAL_ERROR "SentencePiece library target not found")
endif()

target_link_libraries(libgemma ${GEMMA_HWY_LIBS} ${GEMMA_SENTENCEPIECE_LIB})
target_include_directories(libgemma PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(libgemma PRIVATE $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:Clang>>:-Wno-deprecated-declarations>)

# ----------------------------------------------------------------------------
# Backend detection & integration (initial focus: SYCL/oneAPI)
# ----------------------------------------------------------------------------
set(GEMMA_BACKEND_LIBS "")

if(GEMMA_ENABLE_AUTO_BACKENDS)
  # If auto enabled, turn on specific backends unless explicitly OFF
  if(NOT DEFINED GEMMA_ENABLE_CUDA)
    set(GEMMA_ENABLE_CUDA ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_SYCL)
    set(GEMMA_ENABLE_SYCL ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_OPENCL)
    set(GEMMA_ENABLE_OPENCL ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_VULKAN)
    set(GEMMA_ENABLE_VULKAN OFF) # heavier dependency; default off
  endif()
endif()

if(GEMMA_ENABLE_SYCL)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
  find_package(IntelSYCL QUIET)

  # Determine candidate oneAPI roots (priority order):
  set(_ONEAPI_ROOT_CANDIDATES "")
  if(GEMMA_ONEAPI_ROOT)
    list(APPEND _ONEAPI_ROOT_CANDIDATES "${GEMMA_ONEAPI_ROOT}")
  endif()
  if(DEFINED oneapi_root AND NOT oneapi_root STREQUAL "")
    list(APPEND _ONEAPI_ROOT_CANDIDATES "${oneapi_root}")
  endif()
  foreach(_env IN ITEMS GEMMA_ONEAPI_ROOT ONEAPI_ROOT INTEL_ONEAPI_ROOT)
    if(DEFINED ENV{${_env}})
      list(APPEND _ONEAPI_ROOT_CANDIDATES "$ENV{${_env}}")
    endif()
  endforeach()
  # Fallback conventional install path
  list(APPEND _ONEAPI_ROOT_CANDIDATES "C:/Program Files (x86)/Intel/oneAPI")

  list(REMOVE_DUPLICATES _ONEAPI_ROOT_CANDIDATES)

  if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
    message(STATUS "[SYCL][Diag] Candidate oneAPI roots (pre-scan):")
    foreach(_cand ${_ONEAPI_ROOT_CANDIDATES})
      message(STATUS "  - ${_cand}")
    endforeach()
    message(STATUS "[SYCL][Diag] Env snapshot: ONEAPI_ROOT='$ENV{ONEAPI_ROOT}' INTEL_ONEAPI_ROOT='$ENV{INTEL_ONEAPI_ROOT}' GEMMA_ONEAPI_ROOT='${GEMMA_ONEAPI_ROOT}'")
  endif()

  # Manual fallback detection on Windows only sets IntelSYCL_FOUND if ALL components are located
  if(NOT IntelSYCL_FOUND AND WIN32)
    foreach(_cand_root ${_ONEAPI_ROOT_CANDIDATES})
      if(NOT EXISTS "${_cand_root}")
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Skipping non-existent root: ${_cand_root}")
        endif()
        continue()
      endif()
      set(_oneapi_compiler_root "${_cand_root}/compiler")
      if(NOT EXISTS "${_oneapi_compiler_root}")
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Missing compiler dir under: ${_cand_root}")
        endif()
        continue()
      endif()
      file(GLOB _oneapi_version_dirs RELATIVE "${_oneapi_compiler_root}" "${_oneapi_compiler_root}/*")
      set(_version_list "")
      foreach(d ${_oneapi_version_dirs})
        if(d STREQUAL "latest")
          list(APPEND _version_list ${d})
        elseif(d MATCHES "^[0-9]+\\.[0-9]+")
          list(APPEND _version_list ${d})
        endif()
      endforeach()
      set(_sorted_versions "")
      foreach(v ${_version_list})
        if(v STREQUAL "latest")
          list(INSERT _sorted_versions 0 ${v})
        else()
          list(APPEND _sorted_versions ${v})
        endif()
      endforeach()
      foreach(v ${_sorted_versions})
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Scanning version '${v}' under ${_cand_root}")
        endif()
        set(_bin_paths
          "${_oneapi_compiler_root}/${v}/windows/bin"
          "${_oneapi_compiler_root}/${v}/bin"
        )
        set(_include_paths
          "${_oneapi_compiler_root}/${v}/windows/include"
          "${_oneapi_compiler_root}/${v}/include"
        )
        set(_lib_paths
          "${_oneapi_compiler_root}/${v}/windows/lib"
          "${_oneapi_compiler_root}/${v}/lib"
        )
        foreach(_bp ${_bin_paths})
          if(NOT IntelSYCL_COMPILER)
            foreach(_cand icx.exe icpx.exe)
              if(EXISTS "${_bp}/${_cand}")
                set(IntelSYCL_COMPILER "${_bp}/${_cand}")
                break()
              endif()
            endforeach()
          endif()
        endforeach()
        foreach(_ip ${_include_paths})
          if(NOT IntelSYCL_INCLUDE_DIR AND EXISTS "${_ip}/sycl/sycl.hpp")
            set(IntelSYCL_INCLUDE_DIR "${_ip}")
          endif()
        endforeach()
        foreach(_lp ${_lib_paths})
          if(NOT IntelSYCL_LIBRARY)
            file(GLOB _sycl_libs LIST_DIRECTORIES FALSE "${_lp}/sycl*.lib")
            if(_sycl_libs)
              set(_chosen "")
              foreach(_lib ${_sycl_libs})
                if(_lib MATCHES "sycl-devicelib-host")
                  if(NOT _chosen)
                    set(_chosen ${_lib})
                  endif()
                else()
                  set(_chosen ${_lib})
                  break()
                endif()
              endforeach()
              if(_chosen)
                set(IntelSYCL_LIBRARY "${_chosen}")
              endif()
            endif()
          endif()
        endforeach()
        if(IntelSYCL_COMPILER AND IntelSYCL_INCLUDE_DIR AND IntelSYCL_LIBRARY)
          # Infer INTEL_ONEAPI_ROOT if not set yet
          if(NOT INTEL_ONEAPI_ROOT)
            get_filename_component(_comp_dir ${IntelSYCL_COMPILER} DIRECTORY)
            get_filename_component(INTEL_ONEAPI_ROOT ${_comp_dir}/../../.. ABSOLUTE)
          endif()
          set(IntelSYCL_FOUND TRUE)
          set(GEMMA_ONEAPI_ROOT "${INTEL_ONEAPI_ROOT}" CACHE PATH "Detected oneAPI root (propagated)" FORCE)
          message(STATUS "Intel oneAPI SYCL fallback success from root '${_cand_root}':\n  Compiler: ${IntelSYCL_COMPILER}\n  Include : ${IntelSYCL_INCLUDE_DIR}\n  Library : ${IntelSYCL_LIBRARY}")
          if(IntelSYCL_LIBRARY MATCHES "sycl-devicelib-host")
            message(WARNING "Using device library variant (${IntelSYCL_LIBRARY}); linking may require full SYCL import library in some configurations")
          endif()
          break()
        endif()
      endforeach()
      if(IntelSYCL_FOUND)
        break()
      endif()
    endforeach()
    if(NOT IntelSYCL_FOUND)
      message(STATUS "SYCL fallback: unable to locate complete toolchain in candidates: ${_ONEAPI_ROOT_CANDIDATES}")
      if(NOT IntelSYCL_COMPILER)
        message(STATUS "  Missing compiler (icx/icpx)")
      endif()
      if(NOT IntelSYCL_INCLUDE_DIR)
        message(STATUS "  Missing headers (sycl/sycl.hpp)")
      endif()
      if(NOT IntelSYCL_LIBRARY)
        message(STATUS "  Missing import library (sycl*.lib)")
      endif()
      if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
        message(STATUS "[SYCL][Diag] Fallback search exhausted without success.")
      endif()
    endif()
  endif()

  if(IntelSYCL_FOUND)
    message(STATUS "Enabling SYCL backend")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/sycl ${CMAKE_BINARY_DIR}/backends_sycl)
    if(TARGET gemma_sycl_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_sycl_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_SYCL)
    else()
      message(WARNING "SYCL backend sources did not produce target 'gemma_sycl_backend'")
    endif()
  else()
    message(STATUS "Intel SYCL not fully detected; disabling SYCL backend")
  endif()
endif()

if(GEMMA_ENABLE_CUDA)
  enable_language(CXX) # ensure
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found: enabling CUDA backend")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/cuda ${CMAKE_BINARY_DIR}/backends_cuda)
    if(TARGET gemma_cuda_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_cuda_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_CUDA)
    endif()
  else()
    message(STATUS "CUDA toolkit not found; CUDA backend disabled")
  endif()
endif()

if(GEMMA_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/vulkan ${CMAKE_BINARY_DIR}/backends_vulkan)
    if(TARGET gemma_vulkan_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_vulkan_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_VULKAN)
    endif()
  else()
    message(STATUS "Vulkan SDK not found; Vulkan backend disabled")
  endif()
endif()

if(GEMMA_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    # Placeholder: opencl backend directory currently empty; skip add_subdirectory
    message(STATUS "OpenCL found (backend sources missing or not yet implemented)")
    target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_OPENCL)
  else()
    message(STATUS "OpenCL not found; OpenCL backend disabled")
  endif()
endif()

# Apply comprehensive optimizations
apply_gemma_optimizations(libgemma)

install(TARGETS libgemma DESTINATION lib)

# Optionally skip archived sources (they are not in target, but avoid packaging docs)
if(GEMMA_EXCLUDE_ARCHIVE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.archive)
  install(CODE "message(STATUS \"Skipping installation of archived sources\")")
endif()

# Shared library target for C# interop
if(BUILD_GEMMA_DLL)
    add_library(gemma_shared SHARED ${SOURCES})
set_property(TARGET gemma_shared PROPERTY CXX_STANDARD 20)
set_target_properties(gemma_shared PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "gemma"
)
set_property(TARGET gemma_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemma_shared PUBLIC ./ ./gemma)
target_link_libraries(gemma_shared PRIVATE 
  $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy_contrib>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${GEMMA_SENTENCEPIECE_LIB}>
)
target_include_directories(gemma_shared PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(gemma_shared 
    PRIVATE 
    GEMMA_EXPORTS
    $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>
)
target_compile_options(gemma_shared PRIVATE $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:Clang>>:-Wno-deprecated-declarations>)
install(TARGETS gemma_shared DESTINATION lib)
install(FILES gemma/c_api.h DESTINATION include/gemma)
install(FILES gemma/GemmaInterop.cs DESTINATION include/gemma)
endif()

# Executable Target

add_executable(gemma gemma/run.cc)
target_link_libraries(gemma libgemma ${GEMMA_HWY_LIBS} ${GEMMA_BACKEND_LIBS})
apply_gemma_optimizations(gemma)
install(TARGETS gemma DESTINATION bin)

add_executable(single_benchmark evals/benchmark.cc)
target_link_libraries(single_benchmark libgemma ${GEMMA_HWY_LIBS} nlohmann_json::nlohmann_json ${GEMMA_BACKEND_LIBS})

add_executable(benchmarks evals/benchmarks.cc)
target_link_libraries(benchmarks libgemma ${GEMMA_HWY_LIBS} nlohmann_json::nlohmann_json benchmark ${GEMMA_BACKEND_LIBS})

add_executable(debug_prompt evals/debug_prompt.cc)
target_link_libraries(debug_prompt libgemma ${GEMMA_HWY_LIBS} nlohmann_json::nlohmann_json ${GEMMA_BACKEND_LIBS})

## Tests
set(GEMMA_ENABLE_TESTS OFF CACHE BOOL "Enable Gemma tests")
if (GEMMA_ENABLE_TESTS)

enable_testing()
include(GoogleTest)

set(GEMMA_TEST_FILES
  compression/compress_test.cc
  compression/distortion_test.cc
  compression/nuq_test.cc
  compression/sfp_test.cc
  evals/gemma_test.cc
  gemma/tensor_info_test.cc
  io/blob_store_test.cc
  io/fields_test.cc
  ops/bench_matmul.cc
  ops/dot_test.cc
  ops/gemma_matvec_test.cc
  ops/matmul_test.cc
  ops/ops_test.cc
  paligemma/image_test.cc
  paligemma/paligemma_test.cc
  util/threading_test.cc
)

foreach (TESTFILE IN LISTS GEMMA_TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  # Test all targets, not just the best/baseline. This changes the default
  # policy to all-attainable; note that setting -DHWY_COMPILE_* directly can
  # cause compile errors because only one may be set, and other CMakeLists.txt
  # that include us may set them.
  target_compile_options(${TESTNAME} PRIVATE -DHWY_IS_TEST=1)

  target_link_libraries(${TESTNAME} PRIVATE libgemma GTest::Main ${GEMMA_HWY_LIBS} hwy_test)

  gtest_discover_tests(${TESTNAME})
endforeach ()

add_executable(gemma_batch_bench evals/gemma_batch_bench.cc)
target_link_libraries(gemma_batch_bench libgemma GTest::Main ${GEMMA_HWY_LIBS} nlohmann_json::nlohmann_json)

endif()  # GEMMA_ENABLE_TESTS

## Enhanced Testing Framework
# Option to enable the comprehensive test suite
option(GEMMA_ENABLE_ENHANCED_TESTS "Enable enhanced testing framework with unit, integration, and performance tests" OFF)

if(GEMMA_ENABLE_ENHANCED_TESTS)
    message(STATUS "Enhanced testing framework enabled")
    add_subdirectory(../tests tests_build)
endif()

## Tools

add_executable(migrate_weights io/migrate_weights.cc)
target_link_libraries(migrate_weights libgemma ${GEMMA_HWY_LIBS})

# Optional MCP server component
if(GEMMA_ENABLE_MCP)
  if(EXISTS "${CMAKE_SOURCE_DIR}/mcp/CMakeLists.txt")
    message(STATUS "Configuring MCP server component")
    add_subdirectory(${CMAKE_SOURCE_DIR}/mcp mcp_build)
  else()
    message(WARNING "GEMMA_ENABLE_MCP=ON but mcp/ directory missing; skipping")
  endif()
endif()

# ----------------------------------------------------------------------------
# Build Summary Banner
# ----------------------------------------------------------------------------
message(STATUS "\n========== Gemma Build Configuration ==========")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System Deps Preferred: ${GEMMA_PREFER_SYSTEM_DEPS}")
message(STATUS "Griffin Disabled: ${GEMMA_DISABLE_GRIFFIN}")
message(STATUS "Archive Excluded: ${GEMMA_EXCLUDE_ARCHIVE}")
message(STATUS "Auto Backend Detect: ${GEMMA_ENABLE_AUTO_BACKENDS}")
message(STATUS "SYCL Enabled (requested): ${GEMMA_ENABLE_SYCL}")
message(STATUS "CUDA Enabled (requested): ${GEMMA_ENABLE_CUDA}")
message(STATUS "Vulkan Enabled (requested): ${GEMMA_ENABLE_VULKAN}")
message(STATUS "OpenCL Enabled (requested): ${GEMMA_ENABLE_OPENCL}")
if(GEMMA_BACKEND_LIBS)
  string(REPLACE ";" ", " _BACKENDS_LIST "${GEMMA_BACKEND_LIBS}")
  message(STATUS "Active Backend Libraries: ${_BACKENDS_LIST}")
else()
  message(STATUS "Active Backend Libraries: (none)")
endif()
message(STATUS "===============================================\n")

# Print effective compiler flags for visibility (single-config generators)
if(NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Compiler Flags (effective):")
  message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt_uc)
  message(STATUS "  CMAKE_CXX_FLAGS_${_bt_uc}: ${CMAKE_CXX_FLAGS_${_bt_uc}}")
else()
  message(STATUS "Multi-config generator detected; key config flags:")
  foreach(_cfg IN ITEMS Debug Release RelWithDebInfo RelWithSymbols FastDebug MinSizeRel)
    string(TOUPPER "${_cfg}" _ucfg)
    if(DEFINED CMAKE_CXX_FLAGS_${_ucfg})
      message(STATUS "  ${_cfg}: ${CMAKE_CXX_FLAGS_${_ucfg}}")
    endif()
  endforeach()
endif()

# Include dependency summary module (created if not present at top-level)
set(_DEP_SUMMARY_MODULE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DependencySummary.cmake")
if(EXISTS ${_DEP_SUMMARY_MODULE})
  include(${_DEP_SUMMARY_MODULE})
  gemma_print_dependency_summary()
else()
  message(STATUS "(DependencySummary.cmake missing; skipping dependency provenance block)")
endif()

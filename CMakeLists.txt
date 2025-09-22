# Enhanced Gemma.cpp Project Root CMakeLists.txt
# This file orchestrates the build of the enhanced Gemma.cpp with hardware acceleration backends

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Early vcpkg toolchain integration (must occur before first project()).
# If the user did not specify -DCMAKE_TOOLCHAIN_FILE and VCPKG_ROOT is present,
# auto-wire the vcpkg toolchain so that all subsequent find_package() calls
# (in root and subdirectories) resolve to vcpkg-provided ports before any
# FetchContent fallbacks execute in subprojects.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(_VCPKG_CAND "$ENV{VCPKG_ROOT}")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
        # Monorepo layout sibling
        get_filename_component(_VCPKG_CAND "${CMAKE_CURRENT_LIST_DIR}/../vcpkg" ABSOLUTE)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        # In-tree vcpkg clone
        get_filename_component(_VCPKG_CAND "${CMAKE_CURRENT_LIST_DIR}/vcpkg" ABSOLUTE)
    endif()
    if(_VCPKG_CAND AND EXISTS "${_VCPKG_CAND}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_VCPKG_CAND}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain")
        set(GEMMA_USING_VCPKG ON CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
        message(STATUS "[vcpkg] Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    else()
        set(GEMMA_USING_VCPKG OFF CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    endif()
else()
    # Respect user-specified toolchain; detect if it is vcpkg for reporting.
    if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake$")
        set(GEMMA_USING_VCPKG ON CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    else()
        set(GEMMA_USING_VCPKG OFF CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    endif()
endif()

# Set policy versions to handle compatibility with dependencies
# CMP0111: An imported target with INTERFACE_SOURCES may not be built by the generator
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# CMP0091: MSVC runtime library flags are selected by an abstraction
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# CMP0126: set(CACHE) does not remove a normal variable of the same name
if(POLICY CMP0126)
  cmake_policy(SET CMP0126 NEW)
endif()

# CMP0169: FetchContent_Populate deprecation
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 NEW)
endif()

# Set minimum policy version for all subdirectories to handle old dependencies
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # option() honors normal variables
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)  # Include file check macros honor CMAKE_REQUIRED_LIBRARIES

project(gemma_enhanced
    VERSION 1.0.0
    DESCRIPTION "Enhanced Gemma.cpp with MCP server and hardware acceleration backends"
    LANGUAGES CXX C
)

# Early environment validation (optional step to diagnose user setups)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnvValidation)
gemma_validate_environment()

# Set global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Normalize / support custom (non-standard) build types used by presets.
# CMake recognizes: Debug, Release, RelWithDebInfo, MinSizeRel by default.
# We introduce mapping for 'FastDebug' (O1 + debug info) and 'RelWithSymbols'
# (O2 + debug info) so users get expected optimization levels instead of
# falling back to an unrecognized build type with no flags.
# ---------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES) # Single-config generators only
    if(CMAKE_BUILD_TYPE STREQUAL "FastDebug")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(CMAKE_CXX_FLAGS_FASTDEBUG "/Od /O1 /Zi" CACHE STRING "Flags for FastDebug" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_FASTDEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING "Linker flags FastDebug" FORCE)
        else()
            set(CMAKE_CXX_FLAGS_FASTDEBUG "-O1 -g" CACHE STRING "Flags for FastDebug" FORCE)
        endif()
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithSymbols")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "/O2 /Zi /DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_RELWITHSYMBOLS "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}" CACHE STRING "Linker flags RelWithSymbols" FORCE)
        else()
            set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
        endif()
    endif()
else()
    # Multi-config generator (e.g. Visual Studio): extend configuration list if missing.
    set(_EXTRA_CONFIGS_ADDED FALSE)
    foreach(_cfg IN ITEMS FastDebug RelWithSymbols)
        list(FIND CMAKE_CONFIGURATION_TYPES ${_cfg} _idx)
        if(_idx EQUAL -1)
            list(APPEND CMAKE_CONFIGURATION_TYPES ${_cfg})
            set(_EXTRA_CONFIGS_ADDED TRUE)
        endif()
    endforeach()
    if(_EXTRA_CONFIGS_ADDED)
        list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Supported build configurations" FORCE)
        message(STATUS "Added custom build configurations: FastDebug, RelWithSymbols")
    endif()
    # Define per-config flags if not already defined by toolchain files.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS_FASTDEBUG "/Od /O1 /Zi" CACHE STRING "Flags for FastDebug" FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "/O2 /Zi /DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
    else()
        set(CMAKE_CXX_FLAGS_FASTDEBUG "-O1 -g" CACHE STRING "Flags for FastDebug" FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
    endif()
endif()

# ----------------------------------------------------------------------------
# oneAPI root auto-detection (user may supply -DGEMMA_ONEAPI_ROOT= or -Doneapi_root=)
# Propagate as cache variable so subproject backend detection can leverage it.
# ----------------------------------------------------------------------------
if(NOT DEFINED GEMMA_ONEAPI_ROOT OR "${GEMMA_ONEAPI_ROOT}" STREQUAL "")
    if(DEFINED oneapi_root AND EXISTS "${oneapi_root}")
        set(GEMMA_ONEAPI_ROOT "${oneapi_root}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{GEMMA_ONEAPI_ROOT} AND EXISTS "$ENV{GEMMA_ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{GEMMA_ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{INTEL_ONEAPI_ROOT} AND EXISTS "$ENV{INTEL_ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{INTEL_ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    endif()
endif()
if(GEMMA_ONEAPI_ROOT)
    message(STATUS "Detected oneAPI root: ${GEMMA_ONEAPI_ROOT}")
endif()

# Build options for new components
option(GEMMA_BUILD_MCP_SERVER "Build MCP server component" ON)
option(GEMMA_BUILD_BACKENDS "Build hardware acceleration backends" ON)

# Hardware backend options with automatic detection
option(GEMMA_BUILD_SYCL_BACKEND "(Legacy root control) Build SYCL/Intel oneAPI backend" OFF)
option(GEMMA_BUILD_CUDA_BACKEND "(Legacy root control) Build CUDA acceleration backend" OFF)
option(GEMMA_BUILD_OPENCL_BACKEND "(Legacy root control) Build OpenCL acceleration backend" OFF)
option(GEMMA_BUILD_VULKAN_BACKEND "(Legacy root control) Build Vulkan acceleration backend" OFF)
option(GEMMA_BUILD_METAL_BACKEND "(Legacy root control) Build Metal acceleration backend" OFF)
option(GEMMA_USE_LEGACY_BACKEND_BUILD "Use root-level backend detection/add_subdirectory (deprecated). OFF lets gemma.cpp handle backends." OFF)

# Test and documentation options
option(GEMMA_BUILD_ENHANCED_TESTS "Build enhanced test suite" ON)
option(GEMMA_BUILD_BACKEND_TESTS "Build backend-specific tests" ON)
option(GEMMA_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(GEMMA_BUILD_DOCS "Build documentation" OFF)
option(GEMMA_ENABLE_DOXYGEN "Generate Doxygen API docs (implies GEMMA_BUILD_DOCS)" OFF)

# Auto-detection options
option(GEMMA_AUTO_DETECT_BACKENDS "Automatically detect available backends" ON)
option(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS "Print extended backend environment diagnostics during configure" OFF)

# Propagate verbose diagnostics flag to subprojects
set(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS ${GEMMA_VERBOSE_BACKEND_DIAGNOSTICS} CACHE BOOL "Verbose backend diagnostics (propagated)" FORCE)

# Platform detection
if(WIN32)
    set(GEMMA_PLATFORM "Windows")
elseif(APPLE)
    set(GEMMA_PLATFORM "macOS")
elseif(UNIX)
    set(GEMMA_PLATFORM "Linux")
endif()

message(STATUS "Building Enhanced Gemma.cpp for ${GEMMA_PLATFORM}")
message(STATUS "MCP Server: ${GEMMA_BUILD_MCP_SERVER}")
message(STATUS "Hardware Backends: ${GEMMA_BUILD_BACKENDS}")

# Include directories for enhanced components
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Create cmake directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Auto-detect available backends if enabled
if(GEMMA_USE_LEGACY_BACKEND_BUILD AND GEMMA_BUILD_BACKENDS)
    message(STATUS "(LEGACY) Root-level auto-detecting hardware backends...")
    # Legacy path retained for compatibility; modern path handled in gemma.cpp/CMakeLists.txt.
    if(GEMMA_AUTO_DETECT_BACKENDS)
        find_package(IntelSYCL QUIET)
        if(IntelSYCL_FOUND OR EXISTS "$ENV{ONEAPI_ROOT}")
            message(STATUS "Intel oneAPI/SYCL detected (legacy root)")
            set(GEMMA_BUILD_SYCL_BACKEND ON CACHE BOOL "Build SYCL backend" FORCE)
        endif()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit detected (legacy root) ${CUDAToolkit_VERSION}")
            set(GEMMA_BUILD_CUDA_BACKEND ON CACHE BOOL "Build CUDA backend" FORCE)
        endif()
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            message(STATUS "Vulkan SDK detected (legacy root) ${Vulkan_VERSION}")
            set(GEMMA_BUILD_VULKAN_BACKEND ON CACHE BOOL "Build Vulkan backend" FORCE)
        endif()
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL detected (legacy root)")
            set(GEMMA_BUILD_OPENCL_BACKEND ON CACHE BOOL "Build OpenCL backend" FORCE)
        endif()
        if(APPLE)
            message(STATUS "Metal backend available on macOS (legacy root)")
            set(GEMMA_BUILD_METAL_BACKEND ON CACHE BOOL "Build Metal backend" FORCE)
        endif()
    endif()
else()
    message(STATUS "Using embedded gemma.cpp backend detection (root legacy backend detection disabled)")
endif()

# Find required packages for enabled backends
# Map legacy root backend toggles to gemma.cpp cache variables (only if legacy path intentionally used)
if(GEMMA_USE_LEGACY_BACKEND_BUILD)
    set(GEMMA_ENABLE_CUDA ${GEMMA_BUILD_CUDA_BACKEND} CACHE BOOL "Forwarded CUDA enable" FORCE)
    set(GEMMA_ENABLE_SYCL ${GEMMA_BUILD_SYCL_BACKEND} CACHE BOOL "Forwarded SYCL enable" FORCE)
    set(GEMMA_ENABLE_VULKAN ${GEMMA_BUILD_VULKAN_BACKEND} CACHE BOOL "Forwarded Vulkan enable" FORCE)
    set(GEMMA_ENABLE_OPENCL ${GEMMA_BUILD_OPENCL_BACKEND} CACHE BOOL "Forwarded OpenCL enable" FORCE)
endif()

# Set policy minimum version to handle legacy dependencies like sentencepiece
# This is required for sentencepiece which has cmake_minimum_required(VERSION 3.1)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Add the original gemma.cpp as a subdirectory
add_subdirectory(gemma.cpp)

# Add MCP server component if enabled
if(GEMMA_BUILD_MCP_SERVER)
    message(STATUS "Adding MCP server component")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp/CMakeLists.txt)
        add_subdirectory(mcp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
        message(WARNING "MCP directory present but missing CMakeLists.txt; skipping MCP server")
    else()
        message(WARNING "MCP directory not found, skipping MCP server")
    endif()
endif()

# Remove duplicate backend add_subdirectory to avoid target redefinition; handled inside gemma.cpp

# Add enhanced testing framework
if(GEMMA_BUILD_ENHANCED_TESTS)
    message(STATUS "Adding enhanced test suite")
    enable_testing()
    add_subdirectory(tests)
endif()

# Add documentation if enabled
if(GEMMA_BUILD_DOCS OR GEMMA_ENABLE_DOXYGEN)
    message(STATUS "Adding documentation build")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs)
        add_subdirectory(docs)
    else()
        message(WARNING "Docs directory not found, skipping documentation")
    endif()
    if(GEMMA_ENABLE_DOXYGEN)
        find_package(Doxygen QUIET)
        if(DOXYGEN_FOUND)
            set(DOXYGEN_GENERATE_HTML YES)
            set(DOXYGEN_GENERATE_LATEX NO)
            set(DOXYGEN_EXTRACT_PRIVATE NO)
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
            add_custom_target(docs_doxygen
                COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating Doxygen documentation" VERBATIM)
        else()
            message(WARNING "Doxygen requested but not found")
        endif()
    endif()
endif()

# Add tools directory (CLI and other utilities)
message(STATUS "Adding tools directory")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    add_subdirectory(tools)
else()
    message(WARNING "Tools directory not found, skipping tools")
endif()

# Create convenience targets
add_custom_target(gemma_all
    DEPENDS gemma libgemma
    COMMENT "Build all core Gemma components"
)

if(GEMMA_BUILD_MCP_SERVER AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
    add_dependencies(gemma_all gemma_mcp_server)
endif()

if(GEMMA_BUILD_BACKENDS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/backends)
    add_dependencies(gemma_all gemma_backends)
endif()

# Add CLI tool to main target
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    add_dependencies(gemma_all tools)
endif()

# Backend-specific convenience targets
if(GEMMA_BUILD_SYCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_sycl COMMENT "Build with SYCL backend")
endif()

if(GEMMA_BUILD_CUDA_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_cuda COMMENT "Build with CUDA backend")
endif()

if(GEMMA_BUILD_VULKAN_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_vulkan COMMENT "Build with Vulkan backend")
endif()

if(GEMMA_BUILD_OPENCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_opencl COMMENT "Build with OpenCL backend")
endif()

if(GEMMA_BUILD_METAL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_metal COMMENT "Build with Metal backend")
endif()

# Installation configuration
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install CACHE PATH "Installation directory")

# Configure compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /W4")
endif()

# Summary
message(STATUS "=== Enhanced Gemma.cpp Build Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Platform: ${GEMMA_PLATFORM}")
if(CMAKE_BUILD_TYPE MATCHES "FastDebug|RelWithSymbols")
    message(STATUS "Using custom build type '${CMAKE_BUILD_TYPE}' (mapped flags applied)")
endif()
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  - Core Gemma.cpp: ON")
message(STATUS "  - MCP Server: ${GEMMA_BUILD_MCP_SERVER}")
message(STATUS "  - Hardware Backends: ${GEMMA_BUILD_BACKENDS}")
message(STATUS "  - Enhanced Tests: ${GEMMA_BUILD_ENHANCED_TESTS}")
message(STATUS "  - Backend Tests: ${GEMMA_BUILD_BACKEND_TESTS}")
message(STATUS "  - Benchmarks: ${GEMMA_BUILD_BENCHMARKS}")
message(STATUS "  - Documentation: ${GEMMA_BUILD_DOCS}")
message(STATUS "  - Doxygen: ${GEMMA_ENABLE_DOXYGEN}")
message(STATUS "")
message(STATUS "Hardware Backends:")
message(STATUS "  - SYCL/Intel oneAPI: ${GEMMA_BUILD_SYCL_BACKEND}")
message(STATUS "  - CUDA: ${GEMMA_BUILD_CUDA_BACKEND}")
message(STATUS "  - Vulkan: ${GEMMA_BUILD_VULKAN_BACKEND}")
message(STATUS "  - OpenCL: ${GEMMA_BUILD_OPENCL_BACKEND}")
message(STATUS "  - Metal: ${GEMMA_BUILD_METAL_BACKEND}")
message(STATUS "===============================================")
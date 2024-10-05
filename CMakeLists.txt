cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER                "clang"         CACHE STRING "Path to the C compiler")
set(CMAKE_C_FLAGS                   "-Wall"         CACHE STRING "C compiler flags")
set(CMAKE_C_FLAGS_DEBUG             "-g"            CACHE STRING "C compiler debug flags")
set(CMAKE_C_FLAGS_MINSIZEREL        "-Os -DNDEBUG"  CACHE STRING "C compiler min size release flags")
set(CMAKE_C_FLAGS_RELEASE           "-O4 -DNDEBUG"  CACHE STRING "C compiler release flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO    "-O2 -g"        CACHE STRING "C compiler flags for release with debug info")
set(CMAKE_C_STANDARD                11              CACHE STRING "C standard to use")
set(CMAKE_C_STANDARD_REQUIRED       ON              CACHE BOOL "Require specified C standard")

set(CMAKE_CXX_COMPILER              "clang++"       CACHE STRING "Path to the C++ compiler")
set(CMAKE_CXX_FLAGS                 "-Wall"         CACHE STRING "C++ compiler flags")
set(CMAKE_CXX_FLAGS_DEBUG           "-g"            CACHE STRING "C++ compiler debug flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL      "-Os -DNDEBUG"  CACHE STRING "C++ compiler min size release flags")
set(CMAKE_CXX_FLAGS_RELEASE         "-O4 -DNDEBUG"  CACHE STRING "C++ compiler release flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g"        CACHE STRING "C++ compiler flags for release with debug info")
set(CMAKE_CXX_STANDARD              20              CACHE STRING "C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED     ON              CACHE BOOL "Require specified C++ standard")

set(CMAKE_AR                        "ar"            CACHE STRING "Archiver program")
set(CMAKE_LINKER                    "ld"            CACHE STRING "Linker program")
set(CMAKE_NM                        "nm"            CACHE STRING "Symbol table viewer")
set(CMAKE_OBJDUMP                   "objdump"       CACHE STRING "Object dump program")
set(CMAKE_RANLIB                    "ranlib"        CACHE STRING "Archive indexer")

project(toy LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_CLANG_TIDY="/usr/bin/clang-tidy-17")

find_package(LLVM REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

find_package(MLIR REQUIRED)
message(STATUS "Found MLIR in ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

include(AddLLVM)
include(HandleLLVMOptions)

include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_INCLUDE_DIR}" "${MLIR_INCLUDE_DIR}")
link_directories("${LLVM_LIBRARY_DIR} ${MLIR_LIBRARY_DIR}")
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "Choose the type of build." FORCE)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    add_definitions(-DTOY_DEBUG)
else()
    add_definitions(-DTOY_RELEASE)
endif()

set(TOY_VERSION_STRING "0.0.1")

include(AddToy)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib)
add_subdirectory(tools)
cmake_minimum_required(VERSION 3.18)

project(syclArena VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED SYCL_COMPILER)
    set(SYCL_COMPILER "AdaptiveCpp")
endif()

if(SYCL_COMPILER STREQUAL "AdaptiveCpp")
    message("-- Chosen SYCL compiler: AdaptiveCpp")
else()
    message("-- Chosen SYCL compiler: Intel oneAPI")
endif()

if(SYCL_COMPILER STREQUAL "AdaptiveCpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
    set(ACPP_STDPAR 0)
else()
    find_package(IntelSYCL REQUIRED)
endif()

option(SYCLARENA_BUILD_EXAMPLES  "Build examples" ON)
option(SYCLARENA_BUILD_TESTS     "Build tests"    ON)

add_library(syclArena STATIC
    src/allocator.cpp
    src/sysinfo.cpp
)

add_sycl_to_target(TARGET syclArena)

target_include_directories(syclArena
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(syclArena PROPERTIES
    OUTPUT_NAME           syclarena
    POSITION_INDEPENDENT_CODE  ON
)

if(SYCLARENA_BUILD_EXAMPLES)
    add_executable(syclArena_usage examples/usage.cpp)
    target_link_libraries(syclArena_usage PRIVATE syclArena)
    add_sycl_to_target(TARGET syclArena_usage)

    add_executable(syclArena_benchmark examples/benchmark.cpp)
    target_link_libraries(syclArena_benchmark PRIVATE syclArena)
    add_sycl_to_target(TARGET syclArena_benchmark)
endif()

if(SYCLARENA_BUILD_TESTS)
    add_executable(syclArena_test tests/test.cpp)
    target_link_libraries(syclArena_test PRIVATE syclArena)
    add_sycl_to_target(TARGET syclArena_test)
endif()

include(GNUInstallDirs)

install(TARGETS syclArena
    EXPORT syclArenaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/syclarena
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT syclArenaTargets
    FILE      syclArenaTargets.cmake
    NAMESPACE syclArena::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syclArena
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/syclArenaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/syclArenaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syclArena
)

# CMakelists.txt for building libdealii-trixi

set(TARGET_SRC
  src/libdealii.cc
)

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.7.1
  HINTS ${deal.II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

deal_ii_initialize_cached_variables()

project(dealii_trixi CXX C)

include_directories(include)

add_library(dealii_trixi SHARED ${TARGET_SRC})

add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug, now type 'make' to build"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release, now type 'make' to build"
)

deal_ii_initialize_cached_variables()

DEAL_II_SETUP_TARGET(dealii_trixi)

target_link_libraries("dealii_trixi")
include(GNUInstallDirs)
install(TARGETS dealii_trixi
   COMPONENT library
)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)
# execute_process(COMMAND python -c "import torch; print(f'{torch.utils.cmake_prefix_path}/')" OUTPUT_VARIABLE CMAKE_PREFIX_PATH)

option(USE_CUDA "Enable CUDA support" OFF)

message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
message("enabled languages: ${ENABLED_LANGUAGES}")
enable_language(CUDA)
get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
message("enabled languages: ${ENABLED_LANGUAGES}")
message("enabled languages: ${enabled_languages}")
message("enabled languages: ${ENABLED_LANGUAGES}")
message("${Torch_DIR}")
# message("${CUDA_TOOLKIT_ROOT_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(
  INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  )
set(
    COMPILE_DEFINITIONS)

set(
  SOURCES
  ${CMAKE_SOURCE_DIR}/example-app.cpp
  )
message("sources: ******* "${SOURCES})


if(USE_CUDA)
  # enable_language(CUDA)
  # add_definitions(-DUSE_CUDA)
  list(
    APPEND
    COMPILE_DEFINITIONS
    USE_CUDA
  )
endif()

set(
  RNNT_SOURCES
  rnnt/macros.cpp
  rnnt/types.cpp
  rnnt/compute.cpp
  rnnt/cpu/compute.cpp
  )

list(
  APPEND
  SOURCES
  ${RNNT_SOURCES}
  )

message("rnnt sources: ======= "${RNNT_SOURCES})
message("sources: ======= "${SOURCES})

set(main example-app)
add_executable(${main} ${SOURCES})
target_include_directories(${main} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_compile_definitions(${main} PRIVATE ${COMPILE_DEFINITIONS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

set(name torchaudio)
add_library(${name} SHARED ${RNNT_SOURCES})
target_include_directories(${name} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${name} "${TORCH_LIBRARIES}")
target_compile_definitions(${name} PRIVATE ${COMPILE_DEFINITIONS})

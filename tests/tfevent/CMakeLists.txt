cmake_minimum_required(VERSION 3.10)
project(TensorBoardWriter)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Proto files
set(PROTO_FILES
    histogram.proto
    summary.proto
    event.proto
)

# Generate proto sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Add executable
add_executable(tensorboard_writer
    tensorboard_writer.cpp
    ${PROTO_SRCS}
)

# Include directories
target_include_directories(tensorboard_writer PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(tensorboard_writer
    ${Protobuf_LIBRARIES}
)

# Print build info
message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")
message(STATUS "Protobuf include dirs: ${Protobuf_INCLUDE_DIRS}")

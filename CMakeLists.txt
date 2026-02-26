cmake_minimum_required(VERSION 3.16)

project(AMD_GUI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt5 Requirements
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS
    Core
    Gui
    Widgets
    REQUIRED
)

# Source files
set(SOURCES
    src/main.cpp
)

# Create executable
add_executable(AMD_GUI ${SOURCES})

# Link Qt libraries
target_link_libraries(AMD_GUI
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
)

# Set output directory
set_target_properties(AMD_GUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Optional: Set compiler flags for optimized build
if(MSVC)
    target_compile_options(AMD_GUI PRIVATE /W4)
else()
    target_compile_options(AMD_GUI PRIVATE -Wall -Wextra -Wpedantic)
endif()

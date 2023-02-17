cmake_minimum_required(VERSION 3.0.0)
project(cpp_lmpc VERSION 0.0.1)

set(CMAKE_CXX_FLAGS "-std=c++17")
FIND_PACKAGE (Eigen3 REQUIRED)

set( CPPLMPC_SOURCES
    src/CppLmpc.cpp
    src/CppControl.cpp
)

set( CPPLMPC_HEADERS
    include/CppLmpc.hpp
    include/CppControl.hpp
)

# static build
add_library(cpplmpc-static STATIC ${CPPLMPC_SOURCES} ${CPPLMPC_HEADERS})
target_include_directories(cpplmpc-static PUBLIC include)
target_link_libraries (cpplmpc-static Eigen3::Eigen)

# dynamic build
add_library(cpplmpc SHARED ${CPPLMPC_SOURCES} ${CPPLMPC_HEADERS})
target_include_directories(cpplmpc PUBLIC include)
target_link_libraries (cpplmpc Eigen3::Eigen)

# example build
add_executable(cpplmpc_example example/example.cpp)
target_link_libraries(cpplmpc_example cpplmpc-static)
target_link_libraries (cpplmpc_example Eigen3::Eigen)
target_include_directories(cpplmpc_example PUBLIC include)

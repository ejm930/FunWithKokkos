cmake_minimum_required(VERSION 3.13)
project(cool)

set(CMAKE_CXX_STANDARD 14)
find_package(Kokkos REQUIRED)
add_executable(cool main.cpp)
target_link_libraries(cool Kokkos::kokkos)

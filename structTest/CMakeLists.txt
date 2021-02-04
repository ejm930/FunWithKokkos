cmake_minimum_required(VERSION 3.9)

#Set the project name
project(structTest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

find_package(Kokkos REQUIRED)

set(EXEC structTest)

INCLUDE_DIRECTORIES(./src)
file(GLOB SOURCES "src/*.cpp")

add_executable(${EXEC} ${SOURCES})


target_link_libraries(${EXEC} Kokkos::kokkos)


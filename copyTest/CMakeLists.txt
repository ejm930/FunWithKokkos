cmake_minimum_required(VERSION 3.9)

#Set the project name
project(copyTest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

find_package(Kokkos REQUIRED)

set(EXEC copyTest)

INCLUDE_DIRECTORIES(./src)
file(GLOB SOURCES "src/*.cpp")

###### Build Executables ######
#INCLUDE_DIRECTORIES(${CABANA_DIR}/include)
#LINK_DIRECTORIES(${CABANA_DIR}/lib)

#INCLUDE_DIRECTORIES(${KOKKOS_DIR}/include)
#LINK_DIRECTORIES(${KOKKOS_DIR}/lib)

add_executable(${EXEC} ${SOURCES})

#target_link_libraries(${EXEC} libcabanacore.a libkokkos.a dl)

target_link_libraries(${EXEC} Kokkos::kokkos)


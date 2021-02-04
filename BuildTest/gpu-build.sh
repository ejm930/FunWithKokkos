#!/bin/bash

rm -r -f build-gpu
mkdir build-gpu
cd build-gpu

cmake -DCMAKE_PREFIX_PATH="/home/erock/ExtraLibs/CajeteDepend/kokkos/gpu-build/install" -DCMAKE_CXX_COMPILER="/home/erock/ExtraLibs/CajeteDepend/kokkos/bin/nvcc_wrapper" -DCMAKE_CXX_EXTENSIONS=OFF  ..

make -j4

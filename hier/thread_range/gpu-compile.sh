#!/bin/bash

ROOT_DIR=$(cd ../.. && pwd)
rm -r -f gpu-build
mkdir gpu-build
cd gpu-build

CXX=../../../kokkos/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH=$ROOT_DIR/kokkos/gpu-build/install ..

make -j4

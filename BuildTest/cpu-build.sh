#!/bin/bash

rm -r -f build-gpu
mkdir build-gpu
cd build-gpu

cmake -DCMAKE_PREFIX_PATH="/home/erock/ExtraLibs/CajeteDepend/kokkos/cpu-build/install" ..

make -j4

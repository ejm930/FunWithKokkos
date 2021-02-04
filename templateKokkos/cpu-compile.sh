#!/bin/bash

ROOT_DIR=$(cd .. && pwd)
rm -r -f cpu-build
mkdir cpu-build
cd cpu-build

cmake -DCMAKE_PREFIX_PATH=$ROOT_DIR/kokkos/cpu-build/install ..

make -j4

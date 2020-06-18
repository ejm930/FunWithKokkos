#!/bin/bash

cd kokkos

KOKKOS_SRC_DIR=$(pwd)

rm -r -f cpu-build
mkdir cpu-build
cd cpu-build

KOKKOS_INSTALL_DIR=./install

$KOKKOS_SRC_DIR/generate_makefile.bash --with-serial --with-openmp --disable-tests --prefix=$KOKKOS_INSTALL_DIR

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 install

cd ..
rm -r -f gpu-build
mkdir gpu-build
cd gpu-build

echo $KOKKOS_INSTALL_DIR
$KOKKOS_SRC_DIR/generate_makefile.bash --with-serial --with-openmp --with-cuda --arch=Pascal61 --with-cuda-options=enable_lambda --disable-tests --compiler=$KOKKOS_SRC_DIR/bin/nvcc_wrapper --prefix=$KOKKOS_INSTALL_DIR

CXX=$KOKKOS_SRC_DIR/bin/nvcc_wrapper cmake -DENABLE_KOKKOS_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..
make -j4 install

#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi

pushd build
    cmake ../src/
    make
popd
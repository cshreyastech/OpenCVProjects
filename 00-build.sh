#!/bin/bash
echo "Building project"

rm -rf ./build
rm -rf ./out
mkdir build
cmake -B build -S .
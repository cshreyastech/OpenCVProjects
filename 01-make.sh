#!/bin/bash
echo "Make project"

rm -rf ./out/bin/*
cmake --build build --config Release
# ./out/bin/submission
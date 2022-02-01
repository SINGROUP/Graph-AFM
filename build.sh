#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
cd $script_dir/src/c

g++ -fPIC -O3 -c matching.cpp &&
g++ -shared matching.o -o lib.so

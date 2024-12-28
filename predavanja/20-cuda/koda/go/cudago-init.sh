#!/usr/bin/env bash

# run as: 
#   source cuda-init.sh 
# or
#   . cuda-init.sh

module load Go
module load CUDA
export CGO_CFLAGS=$(pkg-config --cflags cudart-12.6)
export CGO_LDFLAGS=$(pkg-config --libs cudart-12.6)
export PATH="~/go/bin/:$PATH"

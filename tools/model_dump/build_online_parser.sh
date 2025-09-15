#!/bin/bash

set -euo pipefail

# get binary name from argument
if [[ $# -ne 1 ]]; then
    echo "No binary name provided. Using default: dump_llm_nodes"
    BIN_NAME=dump_llm_nodes
else
    echo "Binary name provided: $1"
    BIN_NAME=$1
fi

# get current script's path
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get ROOT PATH 
ROOT_PATH="$(cd "$SCRIPT_PATH/../../" && pwd)"

cd "$ROOT_PATH"

bazel build tools/model_dump:$BIN_NAME
pwd
if [[ -f $SCRIPT_PATH/$BIN_NAME ]]; then
    rm -f $SCRIPT_PATH/$BIN_NAME 
fi

cp $ROOT_PATH/bazel-bin/tools/model_dump/$BIN_NAME \
    $SCRIPT_PATH/$BIN_NAME

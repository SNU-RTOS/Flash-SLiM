#!/bin/bash

set -euo pipefail

# get current script's path
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get ROOT PATH 
ROOT_PATH="$(cd "$SCRIPT_PATH/../../.." && pwd)"

cd "$ROOT_PATH"

BIN_NAME=dump_llm
bazel build tools/model_dump/online-parser:$BIN_NAME
pwd
if [[ -f $SCRIPT_PATH/$BIN_NAME ]]; then
    rm -f $SCRIPT_PATH/$BIN_NAME 
fi

cp $ROOT_PATH/bazel-bin/tools/model_dump/online-parser/$BIN_NAME \
    $SCRIPT_PATH/$BIN_NAME

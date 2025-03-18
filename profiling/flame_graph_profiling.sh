#!/bin/bash
ROOT=/home/rtos/workspace/ghpark

sudo perf record -F 5000 -g --call-graph=dwarf -- ./minimal ./mobilenetv3_small_100.lamb_in1k.tflite ./dog.jpg
sudo perf script > out.perf
cat out.perf | ${ROOT}/FlameGraph/stackcollapse-perf.pl > out.folded
${ROOT}/FlameGraph/flamegraph.pl out.folded > flamegraph.svg
python3 -m http.server 8888

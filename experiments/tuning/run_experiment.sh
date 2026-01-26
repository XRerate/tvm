#!/bin/bash
export TVM_NDK_CC=/home/spl/android/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++
cd /home/spl/tvm-android
python -u run_artifact.py --dir tuning_logs/opencl_s25 --tracker-host 127.0.0.1 --tracker-port 9190 --tracker-key R3CX80PSH7N --timeout 600 --repeat 20 --number 10 --warmup-runs 1 "$@"


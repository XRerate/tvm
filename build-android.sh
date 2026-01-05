rm -r build-android && mkdir build-android && cd build-android
cmake .. \
        -DUSE_OPENCL=ON \
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_STL=c++_shared \
        -DUSE_CPP_RPC=ON \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON \
        -DTVM_FFI_USE_LIBBACKTRACE=OFF

cd build-android
make -j$(nproc) tvm_rpc

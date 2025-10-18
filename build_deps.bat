mkdir "build/build_deps"
pushd "build/build_deps"
cmake -G "Visual Studio 16 2019" -A x64 -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%CD%/install ../../packages
cmake --build . --config Release --target install
popd
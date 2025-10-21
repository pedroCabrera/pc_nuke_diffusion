mkdir "build_%BACKEND%/deps"
pushd "build_%BACKEND%/deps"
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DBACKEND=%BACKEND% -DCMAKE_INSTALL_PREFIX=%CD%/install ../../packages
cmake --build . --config Release --target install
popd
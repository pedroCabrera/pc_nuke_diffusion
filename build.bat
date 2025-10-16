@echo off
setlocal
set /p RESET=Build folder exists. Reset (delete and recreate)? [y/N]: 

REM Check if build folder exists
if exist build (
    
    if /I "%RESET%"=="y" (
        rmdir /s /q build
        mkdir build
    ) else (
        echo Using existing build folder.
    )
) else (
    mkdir build
)

cd build
cmake -DCMAKE_PREFIX_PATH="C:\Program Files\Nuke15.1v8" -G "Visual Studio 16 2019" -A x64 -DSD_CUDA=ON ..
cmake --build . --config Release
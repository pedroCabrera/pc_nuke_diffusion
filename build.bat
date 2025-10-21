@echo off
setlocal

set "NUKE_14_1=C:/Program Files/Nuke14.1v1"
set "NUKE_15_1=C:/Program Files/Nuke15.1v8"
set "NUKE_15_2=C:/Program Files/Nuke15.2v4"

set "BACKEND=CUDA"

call build_deps.bat

for %%K in (14_1 15_1 15_2) do (
    echo Building %%K
    mkdir "build_%BACKEND%/build_nuke_%%K"
    call set "NUKE_PATH=%%NUKE_%%K%%"
    call echo Using include dir: %%NUKE_PATH%%
    pushd "build_%BACKEND%/build_nuke_%%K"
    echo "build_%BACKEND%/build_nuke_%%K"
    call cmake -DCMAKE_PREFIX_PATH="%%NUKE_PATH%%" -G "Visual Studio 16 2019" -A x64 -DNUKE_VERSION=%%K -DBACKEND=%BACKEND% -DDEPS_INSTALL_DIR=%CD%\build_%BACKEND%\deps\install ../..
    call cmake --build . --config Release --target install
    popd

)



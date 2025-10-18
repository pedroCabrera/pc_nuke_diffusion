@echo off
setlocal
set /p RESET=Build folder exists. Reset (delete and recreate)? [y/N]: 

set "NUKE_14_1=C:/Program Files/Nuke14.1v1"
set "NUKE_15_1=C:/Program Files/Nuke15.1v8"
set "NUKE_15_2=C:/Program Files/Nuke15.2v4"

call build_deps.bat

for %%K in (14_1 15_1 15_2) do (
    echo Building %%K
    
    REM Check if build folder exists
    if exist "build_nuke_%%K" (
        if /I "%RESET%"=="y" (
            rmdir /s /q "build_nuke_%%K"
            mkdir "build_nuke_%%K"
        ) else (
            echo Using existing build folder.
        )
    ) else (
        mkdir "build_nuke_%%K"
    )
    call set "NUKE_PATH=%%NUKE_%%K%%"
    call echo Using include dir: %%NUKE_PATH%%
    pushd "build_nuke_%%K" >nul
    call cmake -DCMAKE_PREFIX_PATH="%%NUKE_PATH%%" -G "Visual Studio 16 2019" -A x64 -DSD_CUDA=ON -DNUKE_VERSION=%%K -DDEPS_INSTALL_DIR=%CD%\build_deps\install ..
    call cmake --build . --config Release --target install
    popd >nul
)



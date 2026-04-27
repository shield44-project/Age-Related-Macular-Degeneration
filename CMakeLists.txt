cmake_minimum_required(VERSION 3.16)

project(AMD_GUI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(WIN32)
    foreach(_qt_env_var IN ITEMS QTDIR Qt5_DIR Qt6_DIR)
        if(DEFINED ENV{${_qt_env_var}})
            list(APPEND CMAKE_PREFIX_PATH "$ENV{${_qt_env_var}}")
        endif()
    endforeach()

    file(GLOB _qt_default_roots LIST_DIRECTORIES true
        "C:/Qt/*/msvc*_64"
        "C:/Qt/*/mingw*_64"
    )
    list(APPEND CMAKE_PREFIX_PATH ${_qt_default_roots})
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS
    Core
    Gui
    Network
    Widgets
)

find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS
    Core
    Gui
    Network
    Widgets
)

set(SOURCES
    src/main.cpp
)

add_executable(AMD_GUI ${SOURCES})

target_link_libraries(AMD_GUI
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Network
    Qt${QT_VERSION_MAJOR}::Widgets
)

target_compile_definitions(AMD_GUI PRIVATE
    AMD_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

set_target_properties(AMD_GUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(AMD_GUI PRIVATE /W4 /permissive-)
else()
    target_compile_options(AMD_GUI PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS AMD_GUI
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
)

install(DIRECTORY backend
    DESTINATION .
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)

install(FILES requirements.txt
    DESTINATION .
)

option(AMD_GUI_DEPLOY_QT "Copy Qt runtime files beside the Windows executable after build." ON)

if(WIN32 AND AMD_GUI_DEPLOY_QT)
    get_target_property(_qt_qmake_executable Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    if(_qt_qmake_executable)
        get_filename_component(_qt_bin_dir "${_qt_qmake_executable}" DIRECTORY)
        find_program(WINDEPLOYQT_EXECUTABLE
            NAMES windeployqt
            HINTS "${_qt_bin_dir}"
        )
    endif()

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET AMD_GUI POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                    --no-translations
                    --no-system-d3d-compiler
                    "$<TARGET_FILE:AMD_GUI>"
            COMMENT "Deploying Qt runtime files with windeployqt"
            VERBATIM
        )
        set_property(TARGET AMD_GUI PROPERTY
            VS_DEBUGGER_ENVIRONMENT "PATH=${_qt_bin_dir};%PATH%"
        )
    else()
        message(WARNING
            "windeployqt was not found. The build will still succeed, but on "
            "Windows the executable may need Qt DLLs on PATH or beside the exe."
        )
    endif()
endif()

# ── Windows installer via CPack / NSIS ──────────────────────────────────────
# Alternative to the Inno Setup path in installer/AMD_GUI.iss.
# Requires NSIS (https://nsis.sourceforge.io) to be installed.
# Usage:  cmake --build <build-dir> --target package  (or run cpack inside it)
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_NAME "AMD_GUI")
    set(CPACK_PACKAGE_VENDOR "shield44-project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "Age-Related Macular Degeneration Detection System")
    set(CPACK_PACKAGE_VERSION "1.0.0")
    set(CPACK_PACKAGE_VERSION_MAJOR "1")
    set(CPACK_PACKAGE_VERSION_MINOR "0")
    set(CPACK_PACKAGE_VERSION_PATCH "0")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "AMD Detection System")
    set(CPACK_NSIS_DISPLAY_NAME "AMD Detection System")
    set(CPACK_NSIS_PACKAGE_NAME "AMD Detection System")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin")
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "AMD_GUI.exe")
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\AMD Detection System.lnk' \
'$INSTDIR\\\\bin\\\\AMD_GUI.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\AMD Detection System.lnk'"
    )
    set(CPACK_PACKAGE_EXECUTABLES "AMD_GUI" "AMD Detection System")
    include(CPack)
endif()

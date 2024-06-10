cmake_minimum_required(VERSION 3.22)

# Find the OpenGL package
find_package(OpenGL REQUIRED)
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Wno-deprecated -pedantic -Wno-vla-extension -DNDEBUG")
    set(LDFLAGS "${LDFLAGS} -O3 -Wall -Wno-deprecated -Werror -pedantic -DNDEBUG")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-deprecated -pedantic -Wno-vla-extension -I/usr/include/opengl -DNDEBUG")
    set(LDFLAGS "${LDFLAGS} -O3 -Wall -Wno-deprecated -Werror -pedantic -I/usr/include/opengl -DNDEBUG")
endif ()

add_executable(fluid_rendering main.cpp)
target_link_libraries(fluid_rendering
        ${OPENGL_LIBRARIES}
        -lGL -lGLU -lglut
)

# Put the executable in the parent directory
set(EXECUTABLE_OUTPUT_PATH ..)

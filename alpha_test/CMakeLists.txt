cmake_minimum_required(VERSION 3.6)
project(alpha_test)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

INCLUDE(CheckCXXSourceCompiles)
#SET(CMAKE_CXX_FLAGS_DEBUG "-O0")

find_package(CGAL REQUIRED COMPONENTS Core)
include( ${CGAL_USE_FILE} )

find_package(Eigen3)
include( ${EIGEN3_USE_FILE} )
#if (EIGEN3_FOUND)
#    include( ${EIGEN3_USE_FILE} )
#endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if(OPENMP_VERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("

OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})!
To get optimal performance for the \"chunk\" algorithm, use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER:
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7

")
endif()

set(SOURCE_FILES gudhi_tools.h phat_tools.h phat_tools.cpp gudhi_tools.cpp utils.h utils.cpp)

add_executable(homology_3d homology.cpp ${SOURCE_FILES})
target_compile_definitions(homology_3d PUBLIC _MY_DIM=3)

add_executable(homology_4d homology.cpp ${SOURCE_FILES})
target_compile_definitions(homology_4d PUBLIC _MY_DIM=4)

add_executable(homology_5d homology.cpp ${SOURCE_FILES})
target_compile_definitions(homology_5d PUBLIC _MY_DIM=5)

add_executable(dist_matrix dist_matrix.cpp ${SOURCE_FILES})

add_executable(rips rips.cpp ${SOURCE_FILES})

add_executable(density_filtration_3d density_filtration.cpp ${SOURCE_FILES})
target_compile_definitions(density_filtration_3d PUBLIC _MY_DIM=3)
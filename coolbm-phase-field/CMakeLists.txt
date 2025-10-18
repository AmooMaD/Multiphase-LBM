# ******************************************************************************
# CooLBM SOFTWARE LIBRARY

# Copyright ©️ 2025 CORIA Lab., CNRS UMR 6614
# Contact: Dr. Mostafa Safdari Shadloo, msshadloo@coria.fr

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In case of use of CooLBM, please properly cite the following articles:
# [1] R. Alamian, A. K. Nayak, M. S. Shadloo, CooLBM: A Collaborative Open-Source Reactive Multi-Phase/Component Simulation Code via Lattice Boltzmann Method, doi.org/10.48550 arXiv.2502.12955.
# [2] R. Alamian, M. Sawaf, C. Stockinger, A. Hadjadj, J. Latt, M. S. Shadloo, Modeling soot filter regeneration process through surface-reactive flow in porous media using iterative lattice Boltzmann method, Energy, 2024 (289), 129980.
# [3] C. Stockinger, A. Raiolo, R. Alamian, A. Hadjadj, U. Nieken, M. S. Shadloo, Lattice Boltzmann simulations of heterogeneous combustion reactions for application in porous media, Engineering Analysis with Boundary Elements, 2024 (166) 105817.
# [4] A. K. Nayak, A. Singh, M. Mesgarpour, M. S. Shadloo, A Numerical Investigation of Particle Deposition on a Substrate, doi.org/10.48550/arXiv.2502.12719.
# ******************************************************************************

cmake_minimum_required(VERSION 3.6)

project(stlbm LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../out)

message(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID} compiler. Compilation command is ${CMAKE_CXX_COMPILER}.")

set (USE_TBB ON)

# Detect TBB and add it to the project
if(USE_TBB)
    find_package(TBB REQUIRED)
    message(STATUS "Found TBB: ${TBB_DIR}")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall")
    #set (USE_TBB ON)
    #set(CMAKE_EXE_LINKER_FLAGS "-ltbb")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -Wextra -Wpedantic")
    #set (USE_TBB ON)
    #set(CMAKE_EXE_LINKER_FLAGS "-ltbb")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    set(CMAKE_CXX_FLAGS "-std=c++17 -ipo -O3 -xHOST -Wall")
    #set (USE_TBB ON)
    #set(CMAKE_EXE_LINKER_FLAGS "-ltbb")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL PGI) # nvc++ is based on the PGI compiler
	set(CMAKE_CXX_FLAGS "-stdpar -O1 -Wall -std=c++17")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS "/Ox /Ot /GS- /GL /std:c++latest")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} is unknown.")
endif()

set(CMAKE_CUDA_FLAGS "-arch=sm_86 -O3")

add_subdirectory(apps)


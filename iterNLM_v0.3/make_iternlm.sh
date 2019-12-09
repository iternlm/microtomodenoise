#!/bin/bash

############# Parameters ################
#########################################

CUDA_COMPUTECAPABILITY=70
NVCC_PATH="/usr/local/cuda-10.0/bin/nvcc"

#########################################
#########################################

mkdir -p "Build"

if test $CUDA_COMPUTECAPABILITY != 0 && test $CUDA_COMPUTECAPABILITY != 00
then
    echo "building iterNLM_v0.3 with GPU support"
    echo "--------------------------------------------------"
    echo "compiling auxiliary.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/auxiliary.d" "Geometry/auxiliary.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/auxiliary.o" "Geometry/auxiliary.cpp"
    echo "compiling hdcommunication.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/hdcommunication.d" "Geometry/hdcommunication.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/hdcommunication.o" "Geometry/hdcommunication.cpp"
    echo "compiling noiselevel.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/noiselevel.d" "Noise/noiselevel.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/noiselevel.o" "Noise/noiselevel.cpp"
    echo "compiling iternlm_cpu.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/iternlm_cpu.d" "Denoiser/iternlm_cpu.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/iternlm_cpu.o" "Denoiser/iternlm_cpu.cpp"
    echo "compiling iternlm_cpu_kernels.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/iternlm_cpu_kernels.d" "Denoiser/iternlm_cpu_kernels.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/iternlm_cpu_kernels.o" "Denoiser/iternlm_cpu_kernels.cpp"
    echo "compiling iternlm_gpu.cu"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/iternlm_gpu.d" "Denoiser/iternlm_gpu.cu"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -x cu -o  "Build/iternlm_gpu.o" "Denoiser/iternlm_gpu.cu"
    echo "compiling iternlm_prepare.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/iternlm_prepare.d" "Denoiser/iternlm_prepare.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/iternlm_prepare.o" "Denoiser/iternlm_prepare.cpp"    
    echo "compiling main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/main.d" "main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/main.o" "main.cpp"
    echo "linking iterNLM"
    $NVCC_PATH --cudart static --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY -link -o  "iterNLM"  ./Build/main.o  ./Build/noiselevel.o  ./Build/auxiliary.o ./Build/hdcommunication.o  ./Build/iternlm_cpu.o ./Build/iternlm_cpu_kernels.o ./Build/iternlm_gpu.o ./Build/iternlm_prepare.o   -ltiff -lgomp
    echo "--------------------------------------------------"
else
    echo "building iterNLM_v0.3 for CPU"
    echo "--------------------------------------------------"
    echo "compiling auxiliary.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Geometry/auxiliary.cpp" -o $PWD"/Build/auxiliary.o"
    echo "compiling hdcommunication.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Geometry/hdcommunication.cpp" -o $PWD"/Build/hdcommunication.o"
    echo "compiling noiselevel.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Noise/noiselevel.cpp" -o $PWD"/Build/noiselevel.o"
    echo "compiling iternlm_cpu.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Denoiser/iternlm_cpu.cpp" -o $PWD"/Build/iternlm_cpu.o"
    echo "compiling iternlm_cpu_kernels.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Denoiser/iternlm_cpu_kernels.cpp" -o $PWD"/Build/iternlm_cpu_kernels.o"
    echo "compiling iternlm_prepare.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Denoiser/iternlm_prepare.cpp" -o $PWD"/Build/iternlm_prepare.o"
    echo "compiling main_cpu.cpp"
    g++ -Wall -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/main_cpu.cpp" -o $PWD"/Build/main.o"
    echo "linking iterNLM"
    g++  -o $PWD/iterNLM $PWD/Build/main.o  $PWD/Build/noiselevel.o  $PWD/Build/auxiliary.o $PWD/Build/hdcommunication.o  $PWD/Build/iternlm_cpu.o $PWD/Build/iternlm_cpu_kernels.o $PWD/Build/iternlm_prepare.o   -ltiff -lgomp
    echo "--------------------------------------------------"
fi

################################################################################################################################################################
################################################################################################################################################################
 


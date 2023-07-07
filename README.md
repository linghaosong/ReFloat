[![DOI](https://zenodo.org/badge/658977766.svg)](https://zenodo.org/badge/latestdoi/658977766)

# ReFloat

ReFloat is a ReRAM based accelerator for accelerating large-scale sparse linear solvers. 
The ReFloat is a new floating-point format which fits the ReRAM crossbars with hardware cost.
We provided the implemnetations of the conjugate gradient (CG) and the 
biconjugate gradient stabilized (BiCGSTAB) solvers for different platforms including Nvidia GPU, CPU, and ReFloat. 
The matrices used in the evaluation are from [SuiteSparse](https://sparse.tamu.edu) collection in the [martrix market 
format](https://math.nist.gov/MatrixMarket/formats.html).

## The input matrix

We provided one sample matrix `crystm03` in the `matrices` directory. To download and decompress all the matrices, run

    cd matrices
    sh download.sh

## The GPU implemnetation

We have tested the GPU implemneation on a Nvidia V100 GPU with CCuSparse (CUDA version 11.7). To compile the GPU code,

    cd gpu
    make

We provided a script to run all the evaluations,

    sh run_gpu.sh

## The CPU implemnetation

We provided the CPU implemneation and the simulation code under the `src` directory. We suggest that your CPU paltform has [OpenMP](https://www.openmp.org) installed. To compile,

    cd src
    make

To run the CPU implementation, 

    cd run/cpu
    sh run_cpu.sh

## The simulation

To run the simulation,

    cd run/refloat
    sh run_refloat.sh

All the scripts can be configured to run CG by setting `SOLVER=0` or BiCG by setting `SOLVER=1`.

[![DOI](https://zenodo.org/badge/658977766.svg)](https://zenodo.org/badge/latestdoi/658977766)

# ReFloat

ReFloat is a ReRAM based accelerator for accelerating large-scale sparse linear solvers. 
The `refloat` format is a new floating-point format which fits the ReRAM crossbars with hardware cost.
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


# For SC'23 Artifact Evaluation:

Step 1. To obtain a copy of the source code 

    git clone https://github.com/linghaosong/ReFloat.git 

Step 2. To download the matrices 

    cd ReFloat/matrices 
    sh download.sh

Step 3. We provided a GPU implementation which requires a Nvidia P100 or similar HBM GPU. It is optional to run the GPU implementation. To compile the GPU code

    cd ReFloat/gpu 
    make

Step 4. To run the GPU baseline and obtain the results

4.1 To obtain the GPU CG execution time

First, edit the script run_gpu.sh to set 

    PRINT_RES= 
    RUN=gpucg

Then, then run the script 
    
    sh run_gpu.sh

4.2 To obtain the GPU BiCG execution time

First, edit the script run_gpu.sh to set 

    PRINT_RES= 
    RUN=gpubicg

Then, then run the script 
    
    sh run_gpu.sh

4.3 To obtain the GPU CG execution residual 

First, edit the script run_gpu.sh to set 

    PRINT_RES=1 
    RUN=gpucg

Then, then run the script 
    
    sh run_gpu.sh

4.4 To obtain the GPU BiCG execution residual

First, edit the script run_gpu.sh to set 

    PRINT_RES=1 
    RUN=gpubicg

Then, then run the script 
    
    sh run_gpu.sh

Step 5. To compile the simulation code

    cd ReFloat/src
    make

5.1 To run the Feinberg baseline with the assumption that the funtionality is the same as FP64, go to

    cd ReFloat/src/run/baselinefc

To run CG, edit the script run_baselinefc.sh to set 

    SOLVER=0

Then run

    sh run_baselinefc.sh

To run BiCG, edit the script run_baselinefc.sh to set 

    SOLVER=1

Then run

    sh run_baselinefc.sh  


5.2 To run the Feinberg baseline with the evaluation of the functional correctness, go to

    cd ReFloat/src/run/baseline

To run CG, edit the script run_baseline.sh to set 

    SOLVER=0

Then run

    sh run_baseline.sh

To run BiCG, edit the script run_baseline.sh to set 

    SOLVER=1

Then run

    sh run_baseline.sh  

5.3 To run the ReFloat, go to

    cd ReFloat/src/run/refloat

To run CG, edit the script run_refloat.sh to set 

    SOLVER=0

Then run

    sh run_refloat.sh

To run BiCG, edit the script run_refloat.sh to set 

    SOLVER=1

Then run

    sh run_refloat.sh  

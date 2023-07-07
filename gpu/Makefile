CUDA_TOOLKIT = $(shell dirname $$(command -v nvcc))/..
INC = -I$(CUDA_TOOLKIT)/include -I../include
LIBS = -lcudart -lcusparse -Xcompiler -fopenmp -lcublas

FLAGS = -w -O3

.PHONY: all lib clean

.DEFAULT_GOAL := all

all: gpucg gpubicg

gpucg: cg_main.cu
	nvcc $(FLAGS) cg_main.cu -o gpucg $(INC) $(LIBS)

gpubicg: bicg_main.cu
	nvcc $(FLAGS) bicg_main.cu -o gpubicg $(INC) $(LIBS)

clean:
	rm gpucg gpubicg
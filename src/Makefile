INC = -I ../include 

GXX_FLAGS = -w -O3 -fopenmp 

.PHONY: all lib clean

.DEFAULT_GOAL := all

all: cgcpu cgrefloat cgbaseline_fc

OBJS=refloat_blas.o refloat_format.o refloat_helper.o refloat_solver.o

refloat_blas.o: refloat_blas.cpp
	g++ $(GXX_FLAGS) -c $< $(INC)

refloat_format.o: refloat_format.cpp
	g++ $(GXX_FLAGS) -c $< $(INC)

refloat_helper.o: refloat_helper.cpp
	g++ $(GXX_FLAGS) -c $< $(INC)

refloat_solver.o: refloat_solver.cpp
	g++ $(GXX_FLAGS) -c $< $(INC)

cgcpu.o: cgcpu.cpp
	g++ $(GXX_FLAGS) -c cgcpu.cpp $(INC)

cgcpu: $(OBJS) cgcpu.o 
	g++ $(GXX_FLAGS) $(OBJS) cgcpu.o -o cgcpu $(INC)

cgrefloat.o: cgrefloat.cpp
	g++ $(GXX_FLAGS) -c cgrefloat.cpp $(INC)

cgrefloat: $(OBJS) cgrefloat.o 
	g++ $(GXX_FLAGS) $(OBJS) cgrefloat.o -o cgrefloat $(INC)

cgbaseline_fc.o: cgbaseline_fc.cpp
	g++ $(GXX_FLAGS) -c cgbaseline_fc.cpp $(INC)

cgbaseline_fc: $(OBJS) cgbaseline_fc.o 
	g++ $(GXX_FLAGS) $(OBJS) cgbaseline_fc.o -o cgbaseline_fc $(INC)

clean:
	rm *.o
	rm cgcpu cgrefloat cgbaseline_fc cgbaseline

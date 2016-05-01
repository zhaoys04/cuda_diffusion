CPP=g++
NVCC=nvcc
LIBS=-lcudart
CFLAGS=-c -Wall
NVCCFLAGS =-c  

all: diffusion

diffusion: main.o diffusion_kernel.o diffusion.o
	$(NVCC) -o diffusion main.o diffusion_kernel.o diffusion.o $(LIBS)

main.o: main.cpp
	$(CPP) $(CFLAGS) main.cpp $(LIBS)

diffusion.o: diffusion.cpp diffusion.h
	$(CPP) $(CFLAGS) diffusion.cpp $(LIBS)

diffusion_kernel.o: diffusion_kernel.cu diffusion_kernel.cuh
	$(NVCC) $(NVCCFLAGS)  diffusion_kernel.cu $(LIBS)

clean:
	rm *.o
	rm diffusion

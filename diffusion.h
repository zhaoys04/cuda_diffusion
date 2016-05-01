#include "diffusion_kernel.cuh"
CHUNK* divide_chunks(int Nx, int Ny, int GPU_COUNT, int* ny);
void init_chunks(CHUNK * chunks, int ny, int Nx, int Ny, double resolution);

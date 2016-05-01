#include <cuda_runtime.h>

typedef struct {
	int GPU_NO;
	double *h_phi; 
	double *d_phi, *d_result;
	int upper_to_send_offset, lower_to_send_offset,  lower_to_receive_offset;
	int ystart, yend;
	cudaStream_t stream;
} CHUNK;


extern "C" void chunk_step(CHUNK * chunks, int chunk_ny, double dt, double resolution, int GPU_COUNT, dim3 blocks, dim3 threads, int Nx, int Ny);
extern "C" void chunk_update(CHUNK * chunks, int chunk_ny, int GPU_COUNT, int Nx, int Ny);

extern "C" void finalize_chunk(CHUNK * chunks, int chunk_ny, int GPU_COUNT, int Nx, int Ny);
extern "C" void deploy_chunk_on_gpu(CHUNK * chunks, int chunk_ny, int GPU_COUNT);


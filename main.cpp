#include "diffusion.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
	double resolution = 30.0;
	double sx,sy;
	double dt,t;
	double runtime = 10.0;
	FILE * f;
	int Nx,Ny;
	int chunk_ny;
	int GPU_COUNT;
	int ty;
	CHUNK * chunks;

	dt = 0.1/resolution/resolution;
	sx = 10; sy = 10;
	Nx = sx * resolution;
	Ny = sy * resolution;
	dim3 threads(32,32);
	dim3 blocks((Nx + 32 - 1) / 32,(Ny + 32 - 1) / 32);

	cudaGetDeviceCount(&GPU_COUNT);
	printf("there are %d GPU\n",GPU_COUNT);
	chunks = divide_chunks(Nx,Ny,GPU_COUNT,&chunk_ny);
	init_chunks(chunks,chunk_ny,Nx,Ny,resolution);
	deploy_chunk_on_gpu(chunks, chunk_ny, GPU_COUNT);
	
	for (t = 0; t < runtime; t = t + dt){
		chunk_step(chunks, chunk_ny, dt, resolution, GPU_COUNT, blocks, threads, Nx, Ny);
		chunk_update(chunks, chunk_ny, GPU_COUNT, Nx, Ny);
		printf("%f/%f\n",t,runtime);
	}
	finalize_chunk(chunks, chunk_ny, GPU_COUNT, Nx, Ny);
	f = fopen("result.bin","wb");
	int am;
	for (int i=0; i< chunk_ny;i++){
		ty = chunks[i].yend - chunks[i].ystart + 1;	
		if (chunk_ny == 1){
			am = fwrite((void *)chunks[i].h_phi, sizeof(double), Nx * ty, f);
			printf("%d\n",am);
			break;
		}
		if (i == 0)
			am = fwrite((void *)chunks[i].h_phi, sizeof(double), Nx * (ty - 1), f);
		else if (i == chunk_ny - 1)
			am = fwrite((void *)(chunks[i].h_phi + Nx), sizeof(double), Nx * (ty - 1), f);
		else
			am = fwrite((void *)(chunks[i].h_phi + Nx), sizeof(double), Nx * (ty - 2), f);
		printf("%d\n",am);
	}
	fclose(f);
}

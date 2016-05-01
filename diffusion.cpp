#include "diffusion.h"
#include <stdlib.h>
#include <stdio.h>

double source(double x, double y){
	if ( x * x + y * y < 1.0)
		return 1;
	else 
		return 0;
};

CHUNK* divide_chunks(int Nx, int Ny, int GPU_COUNT, int* ny){   // only divide the region in y direction
	int chunk_ny;
	int ystart, yend;
	CHUNK* chunks;
	chunk_ny = 3;
	*ny = chunk_ny;
	chunks = (CHUNK*) malloc(chunk_ny * sizeof(CHUNK));
	ystart = 0;
	for ( int i = 0; i < chunk_ny; i++ ){
		yend = ystart + Ny / chunk_ny - 1;
//		chunks[i].GPU_NO =  i % GPU_COUNT;
		if (i == chunk_ny - 1){
			yend = Ny - 1;
			chunks[i].yend = yend;
		}else{
			chunks[i].yend = yend + 1;
		}
		if (i == 0)
			chunks[i].ystart = ystart;
		else
			chunks[i].ystart = ystart - 1;

		printf("chunk (%d): (%d, %d)\n", i, chunks[i].ystart, chunks[i].yend);
		ystart = yend + 1;		
	}
	return chunks;
}

void init_chunks(CHUNK * chunks, int ny, int Nx, int Ny, double resolution){
	int i;
	int x,y;
	int ty;
	for (i = 0; i < ny; i++){
//		cudaSetDevice(chunks[i].GPU_NO);
//		cudaStreamCreate(&(chunks[i].stream));
		ty = chunks[i].yend - chunks[i].ystart + 1;
		chunks[i].h_phi = (double *) malloc(Nx * ty * sizeof(double));
		chunks[i].d_phi = NULL;
		chunks[i].d_result = NULL;
		for (x = 0; x < Nx; x++)
			for (y = 0; y < ty; y++){
				chunks[i].h_phi[y * Nx + x] = source((x - Nx / 2) / resolution, (y+chunks[i].ystart - Ny / 2) / resolution);
		}
		if (i == 0)
			chunks[i].upper_to_send_offset = 0;
		else
			chunks[i].upper_to_send_offset = Nx;

		if (i == ny - 1){
			chunks[i].lower_to_send_offset = (ty - 2) * Nx;
			chunks[i].lower_to_receive_offset = 0;

		}else{
			chunks[i].lower_to_send_offset = (ty - 2) * Nx;
			chunks[i].lower_to_receive_offset = (ty - 1) * Nx;
		}
	}
}


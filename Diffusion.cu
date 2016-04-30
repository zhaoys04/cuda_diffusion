#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
//(d_phi, d_result, dt,1.0/resolution, Nt, Nx, Ny)
__global__ void diffusion_kernel(double *phi, double *result, const double dt, const double d,  const int Nx, const int Ny){
	int i = blockIdx.x * 32 + threadIdx.x;
	int j = blockIdx.y * 32 + threadIdx.y;
	int nij = j * Nx + i;
	int nijr = nij + 1;
	int nijl = nij - 1;
	int niju = nij - Nx;
	int nijd = nij + Nx;
	if ( i < Nx && j < Ny )
		if (i == 0)
			result[nij] = phi[nijr];
		else if (i == Nx - 1)
			result[nij] = phi[nijl];
		else if (j == 0)
			result[nij] = phi[nijd];
		else if (j == Ny - 1)
			result[nij] = phi[niju];
		else
			result[nij] = phi[nij] + dt/d/d*(phi[nijr]+phi[nijl]+phi[niju]+phi[nijd]-4*phi[nij]);
};

__global__ void test_kernel(double *phi, double *result, const int Nx, const int Ny){
	int i = blockIdx.x * 32 + threadIdx.x;
	int j = blockIdx.y * 32 + threadIdx.y;
	int nij = j * Nx + i;
	if (i < Nx && j < Ny)
		result[nij] = phi[nij] + 1.0;
}

double source(double x, double y){
	if ( x * x + y * y < 1.0)
		return 1;
	else 
		return 0;
};

typedef struct {
	int GPU_NO;
	double *h_phi; 
	double *d_phi, *d_result;
	int upper_to_send_offset, lower_to_send_offset,  lower_to_receive_offset;
	int ystart, yend;
	cudaStream_t stream;
} CHUNK;

CHUNK* divide_chunks(int Nx, int Ny, int GPU_COUNT, int* ny){   // only divide the region in y direction
	int chunk_ny;
	int ystart, yend;
	CHUNK* chunks;
	chunk_ny = 4;
	*ny = chunk_ny;
	chunks = (CHUNK*) malloc(chunk_ny * sizeof(CHUNK));
	ystart = 0;
	for ( int i = 0; i < chunk_ny; i++ ){
		yend = ystart + Ny / chunk_ny - 1;
		chunks[i].GPU_NO =  i % GPU_COUNT;
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
		cudaSetDevice(chunks[i].GPU_NO);
		cudaStreamCreate(&(chunks[i].stream));
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

int main(void) {
	cudaError_t err = cudaSuccess;
	double resolution = 30.0;
	double sx,sy;
	double dt,t;
	double runtime = 10.0;
	FILE * f;
	int Nx,Ny;
	int chunk_nx,chunk_ny;
	int GPU_COUNT;
	int ty;
	double * temp;
	double * buf;
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
	int i = 0, j = 0;
	int ln = 0;

	for ( t = 0; t<runtime; t = t + dt){
	i = 0; 
	while (i <  chunk_ny){
		ln = 0;
		for (j = 0; j < GPU_COUNT && i < chunk_ny; j++){
			cudaSetDevice(chunks[i].GPU_NO);
			ty = chunks[i].yend - chunks[i].ystart + 1;
			if ( chunks[i].d_phi == NULL || chunk_ny > GPU_COUNT){
				cudaMalloc((void **)&chunks[i].d_phi, Nx * ty * sizeof(double));
				cudaMalloc((void **)&chunks[i].d_result, Nx * ty * sizeof(double));
				cudaMemcpyAsync(chunks[i].d_phi, chunks[i].h_phi, Nx * ty * sizeof(double), cudaMemcpyHostToDevice,chunks[i].stream);
			}
			diffusion_kernel<<<blocks,threads,0,chunks[i].stream>>>(chunks[i].d_phi,chunks[i].d_result,dt,1.0/resolution,Nx,ty);
			if (chunk_ny > GPU_COUNT)
				cudaMemcpyAsync(chunks[i].h_phi, chunks[i].d_result, Nx * ty * sizeof(double), cudaMemcpyDeviceToHost,chunks[i].stream);
			printf("launch %d on device %d\n", i, chunks[i].GPU_NO);
			i++;
			ln++;
		}
		for (j = i - ln; j < i && j < chunk_ny; j++){
			cudaStreamSynchronize(chunks[j].stream);
			if (chunk_ny > GPU_COUNT){
				cudaFree(chunks[j].d_phi);
				cudaFree(chunks[j].d_result);
			}
			printf("Synchronizing chunks %d\n",j);
		}
	}
	for (i = 0; i < chunk_ny && chunk_ny > GPU_COUNT; i++){
		if (i == 0)
			memcpy((void *) chunks[i+1].h_phi, (const void *)(chunks[i].h_phi + chunks[i].lower_to_send_offset), Nx * sizeof(double));
		else if (i == chunk_ny - 1)
			memcpy((void *) (chunks[i-1].h_phi + chunks[i-1].lower_to_receive_offset), (const void *)(chunks[i].h_phi + chunks[i].upper_to_send_offset), Nx * sizeof(double));
		else {
			memcpy((void *) (chunks[i-1].h_phi + chunks[i-1].lower_to_receive_offset), (const void *)(chunks[i].h_phi + chunks[i].upper_to_send_offset), Nx * sizeof(double));
			memcpy((void *) chunks[i+1].h_phi, (const void *)(chunks[i].h_phi + chunks[i].lower_to_send_offset), Nx * sizeof(double));
		}

	}
	if (chunk_ny == 1){
		temp = chunks[0].d_phi;
		chunks[0].d_phi = chunks[0].d_result;
		chunks[0].d_result = temp;	
	}else{
		buf = (double *) malloc(Nx * sizeof(double));
		for (i = 0; i < chunk_ny && chunk_ny == GPU_COUNT; i++){
			cudaSetDevice(chunks[i].GPU_NO);
			if (i == 0){
				cudaMemcpy(buf, (chunks[i].d_result+chunks[i].lower_to_send_offset), Nx * sizeof(double),cudaMemcpyDeviceToHost);
				cudaSetDevice(chunks[i+1].GPU_NO);
				cudaMemcpy(chunks[i+1].d_result, buf, Nx * sizeof(double),cudaMemcpyHostToDevice);
			}else if (i == chunk_ny - 1){
				cudaMemcpy(buf, (chunks[i].d_result+chunks[i].upper_to_send_offset), Nx * sizeof(double),cudaMemcpyDeviceToHost);
				cudaSetDevice(chunks[i-1].GPU_NO);
				cudaMemcpy((chunks[i-1].d_result+chunks[i-1].lower_to_receive_offset), buf, Nx * sizeof(double),cudaMemcpyHostToDevice);
			}else {
				cudaMemcpy(buf, (chunks[i].d_result+chunks[i].lower_to_send_offset), Nx * sizeof(double),cudaMemcpyDeviceToHost);
				cudaSetDevice(chunks[i+1].GPU_NO);
				cudaMemcpy(chunks[i+1].d_result, buf, Nx * sizeof(double),cudaMemcpyHostToDevice);
				cudaSetDevice(chunks[i].GPU_NO);
				cudaMemcpy(buf, (chunks[i].d_result+chunks[i].upper_to_send_offset), Nx * sizeof(double),cudaMemcpyDeviceToHost);
				cudaSetDevice(chunks[i-1].GPU_NO);
				cudaMemcpy((chunks[i-1].d_result+chunks[i-1].lower_to_receive_offset), buf, Nx * sizeof(double),cudaMemcpyHostToDevice);

			}		
		}
		for (i = 0;i < chunk_ny && chunk_ny == GPU_COUNT; i++){
			temp = chunks[i].d_phi;
			chunks[i].d_phi = chunks[i].d_result;
			chunks[i].d_result = temp;	
		}
	}

		printf("%f/%f\n",t, runtime);
	}
	if (chunk_ny == 1){
		ty = chunks[0].yend - chunks[0].ystart + 1;
		cudaMemcpy(chunks[0].h_phi, chunks[0].d_result, Nx * ty * sizeof(double), cudaMemcpyDeviceToHost);
	}else{
		for (i = 0;i < chunk_ny && chunk_ny == GPU_COUNT; i++){
			cudaSetDevice(chunks[i].GPU_NO);
			ty = chunks[i].yend - chunks[i].ystart + 1;
			cudaMemcpy(chunks[i].h_phi, chunks[i].d_result, Nx * ty * sizeof(double),cudaMemcpyDeviceToHost);
		}
	}
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

	return 0;
}

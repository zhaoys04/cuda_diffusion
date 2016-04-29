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
	int xstart, xend, ystart, yend;
	cudaStream_t stream;
} CHUNK;

CHUNK* divide_chunks(int Nx, int Ny, int GPU_COUNT, int* nx, int* ny){
	int chunk_nx,chunk_ny;
	int xstart, ystart, xend, yend;
	CHUNK* chunks;
	chunk_nx = 3; chunk_ny = 3;
	*nx = chunk_nx; *ny = chunk_ny;
	chunks = (CHUNK*) malloc(chunk_nx * chunk_ny * sizeof(CHUNK));
	xstart = 0;
	for ( int i = 0; i < chunk_nx; i++ ){
		xend = xstart + Nx / chunk_nx - 1;
		if (i == chunk_nx - 1) xend = Nx - 1;
		ystart = 0;
		for ( int j = 0; j < chunk_ny; j++){
			chunks[j * chunk_nx + i].GPU_NO = (j * chunk_nx + i) % GPU_COUNT;
			yend = ystart + Ny / chunk_ny - 1;
			if (j == chunk_ny - 1)
				yend = Ny - 1;
			if (i == 0)
				chunks[j * chunk_nx + i].xstart = xstart;
			else
				chunks[j * chunk_nx + i].xstart = xstart - 1;
			if (j == 0)
				chunks[j * chunk_nx + i].ystart = ystart;
			else
				chunks[j * chunk_nx + i].ystart = ystart - 1;
			if (i == chunk_nx -1)
				chunks[j * chunk_nx + i].xend = xend;
			else
				chunks[j * chunk_nx + i].xend = xend + 1;

			if (j == chunk_ny -1)
				chunks[j * chunk_nx + i].yend = yend;
			else
				chunks[j * chunk_nx + i].yend = yend + 1;

//			printf("chunk (%d, %d): (%d, %d, %d, %d)\n", i, j, chunks[j * chunk_nx + i].xstart, chunks[j * chunk_nx + i].ystart, chunks[j * chunk_nx + i].xend, chunks[j * chunk_nx + i].yend);
			ystart = yend + 1;		
		}
		xstart = xend + 1;
	}
	return chunks;
}

void init_chunks(CHUNK * chunks, int nx, int ny, int Nx, int Ny, double resolution){
	int i;
	int x,y;
	int tx,ty;
	for (i = 0; i < nx * ny; i++){
		cudaSetDevice(chunks[i].GPU_NO);
		cudaStreamCreate(&(chunks[i].stream));
		tx = chunks[i].xend - chunks[i].xstart + 1;
		ty = chunks[i].yend - chunks[i].ystart + 1;
		chunks[i].h_phi = (double *) malloc(tx * ty * sizeof(double));
		chunks[i].d_phi = NULL;
		chunks[i].d_result = NULL;
		for (x = 0; x < tx; x++)
			for (y = 0; y < ty; y++){
				chunks[i].h_phi[y * ty + x] = source((x+chunks[i].xstart - Nx / 2) / resolution, (y+chunks[i].ystart - Ny / 2) / resolution);
		}
	}
}

int main(void) {
	cudaError_t err = cudaSuccess;
	double resolution = 30.0;
	double sx,sy;
	double dt;
	double runtime = 10.0;
	FILE * f;
	int Nx,Ny;
	int chunk_nx,chunk_ny;
	int GPU_COUNT;
	int tx,ty;
	CHUNK * chunks;

	dt = 0.1/resolution/resolution;
	sx = 10; sy = 10;
	Nx = sx * resolution;
	Ny = sy * resolution;
	dim3 threads(32,32);
	dim3 blocks((Nx + 32 - 1) / 32,(Ny + 32 - 1) / 32);

	cudaGetDeviceCount(&GPU_COUNT);
	printf("there are %d GPU\n",GPU_COUNT);
	chunks = divide_chunks(Nx,Ny,GPU_COUNT,&chunk_nx,&chunk_ny);
	init_chunks(chunks,chunk_nx,chunk_ny,Nx,Ny,resolution);
	int i = 0, j = 0;
	int ln = 0;
	while (i < chunk_nx * chunk_ny){
		ln = 0;
		for (j = 0; j < GPU_COUNT && i < chunk_nx * chunk_ny; j++){
/*			cudaSetDevice(chunks[i].GPU_NO);
			tx = chunks[i].xend - chunks[i].xstart + 1;
			ty = chunks[i].yend - chunks[i].ystart + 1;
			cudaMalloc((void **)&chunks[i].d_phi, tx * ty * sizeof(double));
			cudaMalloc((void **)&chunks[i].d_result, tx * ty * sizeof(double));
			cudaMemcpyAsync(chunks[i].d_phi, chunks[i].h_phi, tx * ty * sizeof(double), cudaMemcpyHostToDevice,chunks[i].stream);
			test_kernel<<<blocks,threads,0,chunks[i].stream>>>(chunks[i].d_phi,chunks[i].d_result,tx,ty);
			cudaMemcpyAsync(chunks[i].h_phi, chunks[i].d_result, tx * ty * sizeof(double), cudaMemcpyDeviceToHost,chunks[i].stream);*/
			printf("launch %d on device %d\n", i, chunks[i].GPU_NO);
			i++;
			ln++;
		}
		for (j = i - ln; j < i && j < chunk_nx * chunk_ny; j++){
//			cudaStreamSynchronize(chunks[j].stream);
			printf("Synchronizing chunks %d\n",j);
		}
	}
	f = fopen("result.bin","wb");
	for (int i=0; i< 1;i++){
		tx = chunks[i].xend - chunks[i].xstart + 1;	
		ty = chunks[i].yend - chunks[i].ystart + 1;	
		fwrite((void *)chunks[i].h_phi, sizeof(double), tx * ty, f);
	}
	fclose(f);

/*	double * h_phi = (double *)malloc(Nx * Ny * sizeof(double));
	double * d_phi = NULL;
	double * d_result = NULL;
	double * temp;
	int step = 0;
	err = cudaMalloc((void **)&d_phi, Nx * Ny * sizeof(double));
	if (err != cudaSuccess)
		printf("malloc failed\n");
	err = cudaMalloc((void **)&d_result, Nx * Ny * sizeof(double));
	if (err != cudaSuccess)
		printf("malloc failed\n");
	memset(h_phi, 0, Nx * Ny * sizeof(double));
	for (int x = 0; x < Nx; x++)
		for (int y = 0; y < Ny; y++)
			if (fabs( (x - Nx/2)/resolution ) < 0.2 && fabs((y - Ny/2)/resolution)< 0.2)
				h_phi[y * Nx + x] = 1.0;

	f = fopen("init.bin","wb");
        fwrite((void *) h_phi, sizeof(double), Nx * Ny, f);
	fclose(f);
	dim3 threads(32,32);
	dim3 blocks((Nx + 32 - 1) / 32,(Ny + 32 - 1) / 32);
	
	cudaMemcpy(d_phi, h_phi, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice);
	for (double t = 0; t < runtime; t=t+dt){
		step++ ;
		diffusion_kernel<<<blocks,threads>>>(d_phi, d_result, dt, 1.0/resolution, Nx, Ny);
		cudaDeviceSynchronize();
		temp = d_phi;
		d_phi = d_result;
		d_result = temp;
		if (step % 200 == 0)
			printf("%f/%f\n",t, runtime);
	}
       	cudaMemcpy(h_phi, d_phi, Nx * Ny * sizeof(double), cudaMemcpyDeviceToHost);
	f = fopen("result.bin","wb");
        fwrite((void *) h_phi, sizeof(double), Nx * Ny, f);
	fclose(f);*/
	return 0;
}

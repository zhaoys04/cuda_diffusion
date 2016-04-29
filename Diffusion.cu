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
}
int main(void) {
	cudaError_t err = cudaSuccess;
	double resolution = 40.0;
	double sx,sy;
	double dt;
	double runtime = 10.0;
	FILE * f;
	int Nx,Ny;
	
	dt = 0.1/resolution/resolution;
	sx = 10; sy = 10;
	Nx = sx * resolution;
	Ny = sy * resolution;
	double * h_phi = (double *)malloc(Nx * Ny * sizeof(double));
//	double * h_result = (double *)malloc(Nx * Ny * sizeof(double));
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
	fclose(f);
	return 0;
}

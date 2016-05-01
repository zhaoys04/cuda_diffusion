#include <string.h>
#include <stdlib.h>
#include "diffusion_kernel.cuh"

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
void deploy_chunk_on_gpu(CHUNK * chunks, int chunk_ny, int GPU_COUNT){
	for (int i = 0; i < chunk_ny; i++){
		chunks[i].GPU_NO =  i % GPU_COUNT;
		cudaSetDevice(chunks[i].GPU_NO);
		cudaStreamCreate(&(chunks[i].stream));
	}

}
void update_chunk_one(CHUNK * chunks){     // only use one chunk
	double * temp;
	temp = chunks[0].d_phi;
	chunks[0].d_phi = chunks[0].d_result;
	chunks[0].d_result = temp;	
}
void update_chunk_less(CHUNK * chunks, int chunk_ny, int Nx){     // number of chunks is less than GPU_COUNT
	double *buf, *temp;
	int i;
	buf = (double *) malloc(Nx * sizeof(double));
	for (i = 0; i < chunk_ny; i++){
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

	for (i = 0;i < chunk_ny; i++){
		temp = chunks[i].d_phi;
		chunks[i].d_phi = chunks[i].d_result;
		chunks[i].d_result = temp;	
	}
}
void update_chunk_more(CHUNK * chunks, int chunk_ny, int Nx){     // number of chunks is more than GPU_COUNT
	int i;
	for (i = 0; i < chunk_ny; i++){
		if (i == 0)
			memcpy((void *) chunks[i+1].h_phi, (const void *)(chunks[i].h_phi + chunks[i].lower_to_send_offset), Nx * sizeof(double));
		else if (i == chunk_ny - 1)
			memcpy((void *) (chunks[i-1].h_phi + chunks[i-1].lower_to_receive_offset), (const void *)(chunks[i].h_phi + chunks[i].upper_to_send_offset), Nx * sizeof(double));
		else {
			memcpy((void *) (chunks[i-1].h_phi + chunks[i-1].lower_to_receive_offset), (const void *)(chunks[i].h_phi + chunks[i].upper_to_send_offset), Nx * sizeof(double));
			memcpy((void *) chunks[i+1].h_phi, (const void *)(chunks[i].h_phi + chunks[i].lower_to_send_offset), Nx * sizeof(double));
		}
	}
}

void finalize_chunk_one(CHUNK * chunks, int Nx, int Ny){     // only use one chunk
	cudaMemcpy(chunks[0].h_phi, chunks[0].d_result, Nx * Ny * sizeof(double), cudaMemcpyDeviceToHost);
}
void finalize_chunk_less(CHUNK * chunks, int chunk_ny, int Nx){     // number of chunks is less than GPU_COUNT
	int i, ty;
	for (i = 0;i < chunk_ny; i++){
		cudaSetDevice(chunks[i].GPU_NO);
		ty = chunks[i].yend - chunks[i].ystart + 1;
		cudaMemcpyAsync(chunks[i].h_phi, chunks[i].d_result, Nx * ty * sizeof(double),cudaMemcpyDeviceToHost,chunks[i].stream);
	}
	for (i = 0; i < chunk_ny; i++)
		cudaStreamSynchronize(chunks[i].stream);
}

void chunk_step_one_chunk(CHUNK * chunks, double dt, double resolution, dim3 blocks, dim3 threads, int Nx, int Ny){   // only use one chunk
		cudaSetDevice(chunks[0].GPU_NO);
		if (chunks[0].d_phi != NULL){
			diffusion_kernel<<<blocks,threads>>>(chunks[0].d_phi,chunks[0].d_result,dt,1.0/resolution,Nx,Ny);
		}else{
			cudaMalloc((void **)&chunks[0].d_phi, Nx * Ny * sizeof(double));
			cudaMalloc((void **)&chunks[0].d_result, Nx * Ny * sizeof(double));
			cudaMemcpy(chunks[0].d_phi, chunks[0].h_phi, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice);
			diffusion_kernel<<<blocks,threads>>>(chunks[0].d_phi,chunks[0].d_result,dt,1.0/resolution,Nx,Ny);
		}
}

void chunk_step_less_chunk(CHUNK * chunks, int chunk_ny, double dt, double resolution, dim3 blocks, dim3 threads, int Nx){ // number of chunks is less than or equal to GPU_COUNT
	int i,ty;
	for (i = 0; i < chunk_ny; i++){
		cudaSetDevice(chunks[i].GPU_NO);
		ty = chunks[i].yend - chunks[i].ystart + 1;
		if (chunks[i].d_phi != NULL)
			diffusion_kernel<<<blocks,threads,0,chunks[i].stream>>>(chunks[i].d_phi,chunks[i].d_result,dt,1.0/resolution,Nx,ty);
		else{
			cudaMalloc((void **)&chunks[i].d_phi, Nx * ty * sizeof(double));
			cudaMalloc((void **)&chunks[i].d_result, Nx * ty * sizeof(double));
			cudaMemcpyAsync(chunks[i].d_phi, chunks[i].h_phi, Nx * ty * sizeof(double), cudaMemcpyHostToDevice,chunks[i].stream);
			diffusion_kernel<<<blocks,threads,0,chunks[i].stream>>>(chunks[i].d_phi,chunks[i].d_result,dt,1.0/resolution,Nx,ty);

		}
	}

	for (i = 0; i < chunk_ny; i++)
		cudaStreamSynchronize(chunks[i].stream);

}

void chunk_step_more_chunk(CHUNK * chunks, int chunk_ny, double dt, double resolution, int GPU_COUNT, dim3 blocks, dim3 threads, int Nx){ // number of chunks is greater than GPU_COUNT
	int i, j, ln, ty;
	i = 0; 
	while (i <  chunk_ny){
		ln = 0;
		for (j = 0; j < GPU_COUNT && i < chunk_ny; j++){
			cudaSetDevice(chunks[i].GPU_NO);
			ty = chunks[i].yend - chunks[i].ystart + 1;
			cudaMalloc((void **)&chunks[i].d_phi, Nx * ty * sizeof(double));
			cudaMalloc((void **)&chunks[i].d_result, Nx * ty * sizeof(double));
			cudaMemcpyAsync(chunks[i].d_phi, chunks[i].h_phi, Nx * ty * sizeof(double), cudaMemcpyHostToDevice,chunks[i].stream);
			diffusion_kernel<<<blocks,threads,0,chunks[i].stream>>>(chunks[i].d_phi,chunks[i].d_result,dt,1.0/resolution,Nx,ty);
			cudaMemcpyAsync(chunks[i].h_phi, chunks[i].d_result, Nx * ty * sizeof(double), cudaMemcpyDeviceToHost,chunks[i].stream);
			i++;
			ln++;
		}
		for (j = i - ln; j < i && j < chunk_ny; j++){
			cudaStreamSynchronize(chunks[j].stream);
			cudaFree(chunks[j].d_phi);
			cudaFree(chunks[j].d_result);
		}
	}
}

void chunk_step(CHUNK * chunks, int chunk_ny, double dt, double resolution, int GPU_COUNT, dim3 blocks, dim3 threads, int Nx, int Ny){
	if (chunk_ny == 1)
		chunk_step_one_chunk(chunks, dt, resolution, blocks, threads, Nx, Ny);
	else if (chunk_ny <= GPU_COUNT)
		chunk_step_less_chunk(chunks, chunk_ny, dt, resolution, blocks, threads, Nx);
	else
		chunk_step_more_chunk(chunks, chunk_ny, dt, resolution, GPU_COUNT, blocks, threads, Nx);	
}

void chunk_update(CHUNK * chunks, int chunk_ny, int GPU_COUNT, int Nx, int Ny){
	if (chunk_ny == 1)
		update_chunk_one(chunks);
	else if (chunk_ny <= GPU_COUNT)
		update_chunk_less(chunks, chunk_ny, Nx);
	else
		update_chunk_more(chunks, chunk_ny, Nx);
}

void finalize_chunk(CHUNK * chunks, int chunk_ny, int GPU_COUNT, int Nx, int Ny){
	if (chunk_ny == 1)
		finalize_chunk_one(chunks, Nx, Ny);
	else if (chunk_ny <= GPU_COUNT)
		finalize_chunk_less(chunks, chunk_ny, Nx);
}

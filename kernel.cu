
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {


    // TODO
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_S[TILE_DIM][TILE_DIM];

    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    float sum =0.0f;
    for(unsigned int tile = 0; tile < ((K+blockDim.x-1)/TILE_DIM; ++tile){
        if(tile*TILE_DIM + threadIdx.x < N && row < M){
            A_s[threadIdx.y][threadIdx.x] = A[row*M + tile*TILE_DIM + threadIdx.x];
        }
        else{
            A_s[threadIdx.y][threadIdx.x]=0.0;
        }
        if(k*TILE_DIM + threadIdx.y <M  && col < N){
            B_s[threadIdx.y][threadIdx.x]= B[(tile*TILE_DIM + threadIdx.y)*N + col];
        }else{
            B_s[threadIdx.y][threadIdx.x]=0.0;

        }
            __syncthreads();

            for(unsigned int i = 0; i < TILE_DIM; ++i) {
                    sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
            }
            __syncthreads();

    }
    C[row*N + col] = sum;

}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d, *B_d,*C_d;
    cudaMalloc((void**) &A_d,M*K*sizeof(float));
    cudaMalloc((void**) &B_d,N*K*sizeof(float));
    cudaMalloc((void**) &A_d,M*N*sizeof(float));

    // TODO

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    startTime(&timer);
    cudaMemcpy(A_d, A,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B,N*K*sizeof(float),cudaMemcpyHostToDevice);

    // TODO

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((M+numThreadsPerBlock.x-1)/numThreadsPerBlock.x,(N+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);
    mm_tiled_kernel<<numThreadsPerBlock,numBlocks>> (A_d, B_d, C_d, M, N,  K);

    // TODO

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(C, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}


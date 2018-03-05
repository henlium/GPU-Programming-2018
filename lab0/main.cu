#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void hello_kernel(char *odata, int num) {
    char hello_str[480] = "#######################################\n  _    _                  _            \n | |  | |                | |           \n | |__| | ___ _ __   __ _| | ___  ___  \n |  __  |/ _ \\ '_ \\ / _` | |/ _ \\/ _ \\ \n | |  | |  __/ | | | (_| | |  __/  __/ \n |_|  |_|\\___|_| |_|\\__, |_|\\___|\\___| \n                     __/ |             \n                    |___/              \n+++++++++++++++++++++++++++++++++++++++\n:::::::::::::::::::::::::::::::::::::::\n";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num)
        odata[idx] = hello_str[idx];
}

int main(void) {
    char *h_data, *d_data;
    const int strlen = 480;
    size_t strsize = strlen * sizeof(char);
    h_data = (char *) malloc(strsize);
    memset(h_data, 0, strlen);
    cudaMalloc((void **) &d_data, strsize);
    cudaMemcpy(d_data, h_data, strsize, cudaMemcpyHostToDevice);
    int blocksize = 8;
    int nblock = strlen/blocksize + (strlen % blocksize == 0 ? 0 : 1);

    hello_kernel<<<nblock,blocksize>>>(d_data, strlen);

    cudaMemcpy(h_data, d_data, sizeof(char)*strlen, cudaMemcpyDeviceToHost);
    printf("%s\n", h_data);

    free(h_data);
    cudaFree(d_data);
}
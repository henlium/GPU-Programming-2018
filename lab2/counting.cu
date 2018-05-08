#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <thrust/device_vector.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#define THREAD_SIZE 512

struct to01
{
    __host__ __device__
    int operator()(char c) {if (c == '\n') return 0; return 1;}
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_ptr<const char> d_text(text);
    thrust::device_ptr<int> d_pos(pos);
    thrust::fill(d_pos, d_pos + text_size, 1);
    thrust::device_vector<int> keys(text_size);

    to01 op;
    thrust::transform(d_text, d_text+text_size, keys.begin(), op);
    thrust::inclusive_scan_by_key(thrust::device, keys.begin(), keys.end(), d_pos, d_pos);
    thrust::transform(d_pos, d_pos+text_size, keys.begin(), d_pos, thrust::multiplies<int>());
}

__global__ void Count(const char* text, int* pos, int end)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > end)
		return;
	if (text[idx] == '\n')
		return;

	if (text[idx-1] == '\n' || idx == 0)
	{
        int count = 1;
        while(1)
        {
            pos[idx] = count++;
            if (text[++idx] == '\n' || idx > end) return;
        }
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	Count <<<CeilDiv(text_size, THREAD_SIZE), THREAD_SIZE>>>(text, pos, text_size -1);
}

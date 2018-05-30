#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv( int a, int b ) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign( int a, int b ) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
) {
  const int yt = blockIdx.y * blockDim.y + threadIdx.y;
  const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int n_curt = curt-wt,
						w_curt = curt-1,
						e_curt = curt+1,
						s_curt = curt+wt;
  if ( xt >= 0 and xt < wt-1 and yt > 0 and yt < ht-1 and mask[curt] > 127.0f ) {
    const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		const int n_curb = curb-wb,
							w_curb = curb-1,
							e_curb = curb+1,
							s_curb = curb+wb;
    if ( yb < hb and xb < wb ) {
			int n_px = 4;
			float sur_output[3] = {0.0f, 0.0f, 0.0f},
						sur_target[3] = {0.0f, 0.0f, 0.0f};
			for (int i = 0; i < 3; i++) {
				n_px = 4;
				if (yb > 0) {
					sur_output[i] += output[n_curb*3+i];
					sur_target[i] += target[n_curt*3+i];
				}
				else
					n_px--;
				if (xb > 0) {
					sur_output[i] += output[w_curb*3+i];
					sur_target[i] += target[w_curt*3+i];
				}
				else
					n_px--;
				if (xb < wb-1) {
					sur_output[i] += output[e_curb*3+i];
					sur_target[i] += target[e_curt*3+i];
				}
				else
					n_px--;
				if (yb < hb-1) {
					sur_output[i] += output[s_curb*3+i];
					sur_target[i] += target[s_curt*3+i];
				}
				else
					n_px--;

				output[curb*3+i] = target[curt*3+i] + (sur_output[i]-sur_target[i])/n_px;
			}
		}
	}
}

void PoissonImageCloning(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
)
{
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
  for ( auto i = 0; i < 17754; ++i ) {
    SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
      background, target, mask, output,
      wb, hb, wt, ht, oy, ox
    );
  }
}
#ifndef GPU_CONSTANTS_CUH
#define GPU_CONSTANTS_CUH

#include <iostream>
#include <cuda.h>

namespace denoise
{
	//#define USE_DYNAMIC_MALLOC

	namespace gpu_const
	{
		__constant__ int nx, ny, nz;
		__constant__ int radius_searchspace[3] = {10, 10, 10};
		__constant__ int nslices_searchspace = 11;
		__constant__ int radius_patchspace[3] = {1, 1, 1};
		__constant__ int nsize_search = 0;
		__constant__ int nsize_patch = 0;

		__constant__ float beta = 1.f;

		__constant__ int padding[6] = {0,0,0,0,0,0};
	}
}

#endif //GPU_CONSTANTS_CUH

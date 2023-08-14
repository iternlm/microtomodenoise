#ifndef ITERNLM_PREPARE_H
#define ITERNLM_PREPARE_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include "../denoise_parameters.h"
#include <math.h>

namespace denoise
{
	float* pad_reflective(float* imagestack, int padding[6], const int inshape[3], int outshape[3]);
	float* pad_reflective_unrollpatchspace(float* imagestack, int padding[6], const int inshape[3], int outshape[3], long long int *patchpositions, int nsize_patch);

	long long int* setup_searchspace(int shape[3], protocol::DenoiseParameters *params, int &nsize_search);
	long long int* setup_patchspace(int shape[3], protocol::DenoiseParameters *params, int &nsize_patch);
	float* setup_distweight(int shape[3], protocol::DenoiseParameters *params);

	//experimental stuff:
	float* setup_gaussian_searchweight(float sigma, int shape[3], protocol::DenoiseParameters *params);
}

#endif // ITERNLM_PREPARE_H

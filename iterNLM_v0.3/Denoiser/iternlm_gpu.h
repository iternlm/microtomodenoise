#ifndef ITERNLM_GPU_H
#define ITERNLM_GPU_H

#include <iostream>
#include "../denoise_parameters.h"
#include "iternlm_prepare.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace denoise
{
	typedef long long int idx_type;

    class IterativeNLM_GPU
    {
    public:
		int n_parallelslices = 0;
		bool resumed = false;

    	int configure_device(int shape[3], protocol::DenoiseParameters *params);
    	void free_device();

    	void Run_GaussianNoise(int iter, float* instack, int shape[3], protocol::DenoiseParameters *params);
    	void Run_GaussianNoise_GPUBlocks(int iter, float* instack, float* &previous, int shape[3], protocol::DenoiseParameters *params);
    	void Run_GaussianNoise_GPUBlocks(int iter, std::vector<std::string> &filelist_raw, std::vector<std::string> &filelist_prev, int shape[3], protocol::DenoiseParameters *params);

    	void set_sigma(float* sigmalist, int shape[3]);
    	void get_result(float* output, int shape[3]);

    private:
    	float **image_raw, **image_previous, **next_result;
    	long long int **search_positions, **patch_positions;
    	float **distweight, **sigma_list;

    	int nsize_search, nsize_patch;

    	int deviceID = 0;
    	int ngpus = 1;
    	int threadsPerBlock = 128;

    	int shape_padded[3];
    	int padding[6];

    	long long int expected_filesize = 0;

    	void prepare_iteration1(float* input, int shape[3]);
    	void prepare_iteration1_block(float* input, int shape[3], const int* firstslice, const int* lastslice);
    	void prepare_nextiteration(int shape[3]);
    	void prepare_nextiteration_block(float* input, float *prev_result, int shape[3], const int* firstslice, const int* lastslice);
	};
}

#endif //ITERNLM_GPU_H

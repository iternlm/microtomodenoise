#ifndef DENOISE_PARAMETERS_H
#define DENOISE_PARAMETERS_H

#include <iostream>
#include <string.h>

namespace protocol
{
	struct DenoiseParameters
	{
		float alpha = 0.5f;
		int maxiterations = 4;

		int nslices = 11; //Number of adjacent slices used in denoising.
		int radius_searchspace[3] = {10, 10, 10}; //Radius of ellipsoidal search space. (5 is often better than 10 if there are no artifacts)
		int radius_patchspace[3] = {1, 1, 1}; //Radius of spherical patch space by axis.
		float beta = 1.f; //allows to change the smoothing parameter

		float z_anisotropy = 1.f;

		struct NoiseLevel
		{
			std::string mode = "z-adaptive"; //how to estimate the noise level: "z-adaptive", "global", "manual", "semimanual"
			int zwindow = 100; //for "z-adaptive" mode
			float sigma[2] = {0.0f,0.0f}; //for manual mode

			uint64_t n_samples = 1e5; //how many samples to draw for automatic estimate
			int patchsize = 15; //size of 2D-patch used for noise evaluation

			float stds_from_mean = 1; //shift the noise level by std given
			bool continuous_estimate = false; //estimate more often than twice. (for detail preservation set alpha=0 and increase noiseshift

			int circular_mask_diameter = 0; //when > 0 mask out noise estimate outside circular region
		};
		struct CPU
		{
			int max_threads = 128;

			bool unrolled_patchspace = false; //add a 4th dimension containing patch values to the data (increases read-out speed)
			bool blockwise = true; //work on subsets with the depth of the amount of available threads
		};
		struct GPU
		{
			int n_gpus = 2;
			int deviceID = 0;
			int threadsPerBlock = 128;

			int memory_buffer = 0.1; //relative amount of GPU memory kept free
			bool blockwise_host = false; //read and write blocks from and to disk to save host memory (like blockwise but based on slices that fit in GPU)
		};
		struct IO{
			bool resume = false;
			int firstslice = -1;
			int lastslice = -1;

			std::string save_type = "32bit";
			bool cleanup = false;
			bool rgb = false;

			//internally set:
			std::string active_outpath = "";
		};
		struct Color{
			bool stabilize_variance = true; //apply Anscombe Transform to approximate Poisson noise as Gaussian
			bool independent_channels = false;
			bool average_channelsigma = true; //use the same sigma for all channels
		};

		NoiseLevel noiselevel;
		CPU cpu;
		GPU gpu;
		IO io;
		Color color;
	};
}

#endif //DENOISE_PARAMETERS_H

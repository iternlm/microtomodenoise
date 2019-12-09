#ifndef COLOR_DENOISING_H
#define COLOR_DENOISING_H

#include <iostream>
#include <string.h>
#include <vector>

#include "../Geometry/hdcommunication.h"
#include "../denoise_parameters.h"
#include "../Denoiser/iternlm_cpu.h"
#include "../Noise/noiselevel.h"
#include "../Geometry/auxiliary.h"
#include "colorspace.h"

namespace color_denoise
{
	//Poisson noise will be stabilized with an Anscombe transform (otherwise: --nopoisson)

	void run_colorimage_denoising(std::vector<std::string> &filelist, std::string outpath, protocol::DenoiseParameters *params)
	{
		int shape[3] = {0,0,1};
		hdcom::HdCommunication hdcom;

		float *sigmalist = (float*) malloc(sizeof(*sigmalist));

		//For color images it is assumed that the directory contains 2D images:
		params->radius_patchspace[2] = 0;
		params->radius_searchspace[2] = 0;
		params->nslices = 1;
		params->noiselevel.n_samples = 2048;

		for (int n = 0; n < (int) filelist.size(); n++)
		{
			std::cout << "--------------------------------------" << std::endl;
			std::string filename = filelist[n].substr(filelist[n].rfind("/", filelist[n].length())+1, filelist[n].rfind(".", filelist[n].length())-filelist[n].rfind("/", filelist[n].length())-1);
			std::cout << "image: " << filename << std::endl;

			//Read into separate channels
			//////////////////////////////////////////////////////////////////
			float *R, *G, *B;
			hdcom.GetRGBTif_32bitChannels(filelist[n], shape, R, G, B);
			int nslice = shape[0]*shape[1];
			//////////////////////////////////////////////////////////////////

			//Apply Anscombe Transform and prepare
			//////////////////////////////////////////////////////////////////
			//color::RGB2HSV(R,G,B,shape);

			if(params->color.stabilize_variance) color::anscombe_transform(R,G,B,shape);

			denoise::IterativeNLM_CPU iternlm;
			std::vector<float*> image_raw = {R, G, B};
			std::vector<float*> results;

			uint8_t* output = (uint8_t*) calloc(3*nslice, sizeof(*output));
			//////////////////////////////////////////////////////////////////

			//Estimate noise (will need Poisson)
			//////////////////////////////////////////////////////////////////
			std::vector<float> sigma0, sigma1;
			noise::NoiseLevel noise(params->noiselevel.n_samples, params->noiselevel.patchsize, shape);

			if ("get initial noise level"){
				sigma0 = noise.get_noiselevel_2D_RGB(R, G, B, params);
				std::cout << "initial noise level: " << sigma0[0] << ", " << sigma0[1] << ", " << sigma0[2] << "          " << std::endl;
				if (params->color.average_channelsigma) sigma0[0] = sigma0[1] = sigma0[2] = (sigma0[0]+sigma0[1]+sigma0[2])/3.f;
			}
			//////////////////////////////////////////////////////////////////

			//Denoise with noise estimate from channel
			//////////////////////////////////////////////////////////////////
			iternlm.Run_GaussianNoise_3channels(1, image_raw, results, sigma0, shape, params);

			if(!params->io.cleanup)
			{
				if(params->color.stabilize_variance) color::inverse_anscombe_transform(results[0],results[1],results[2],shape);
				for (int idx = 0; idx < nslice; idx++)
				{
					output[3*idx] = (uint8_t) std::min(255.f, results[0][idx]);
					output[3*idx+1] = (uint8_t) std::min(255.f, results[1][idx]);
					output[3*idx+2] = (uint8_t) std::min(255.f, results[2][idx]);
				}
				if(params->color.stabilize_variance) color::anscombe_transform(results[0],results[1],results[2],shape);
				hdcom.Save2DTifImage_RGB(output, shape, outpath+"/denoise_iterations/", filename+"_iter"+aux::zfill_int2string(1,2));
			}
			//////////////////////////////////////////////////////////////////

			//reestimate and weight the noise level
			/////////////////////////////////////////////////////////////////////
			if ("get texture level" && params->maxiterations > 1){
				sigma1 = noise.get_noiselevel_2D_RGB(results[0], results[1], results[2], params);
				std::cout << "2nd noise level: " << sigma1[0] << ", " << sigma1[1] << ", " << sigma1[2] << "          " << std::endl;
				if (params->color.average_channelsigma) sigma1[0] = sigma1[1] = sigma1[2] = (sigma1[0]+sigma1[1]+sigma1[2])/3.f;

				sigma0[0] = params->alpha*sigma0[0] + (1.f-params->alpha)*sigma1[0];
				sigma0[1] = params->alpha*sigma0[1] + (1.f-params->alpha)*sigma1[1];
				sigma0[2] = params->alpha*sigma0[2] + (1.f-params->alpha)*sigma1[2];
			}
			/////////////////////////////////////////////////////////////////////

			//run remaining denoising iteration
			/////////////////////////////////////////////////////////////////////
			for (int iter = 2; iter <= params->maxiterations; iter++)
			{
				iternlm.Run_GaussianNoise_3channels(iter, image_raw, results, sigma0, shape, params);

				if(!params->io.cleanup)
				{
					if(params->color.stabilize_variance) color::inverse_anscombe_transform(results[0],results[1],results[2],shape);
					for (int idx = 0; idx < nslice; idx++)
					{
						output[3*idx] = (uint8_t) std::min(255.f, results[0][idx]);
						output[3*idx+1] = (uint8_t) std::min(255.f, results[1][idx]);
						output[3*idx+2] = (uint8_t) std::min(255.f, results[2][idx]);
					}
					if(params->color.stabilize_variance) color::anscombe_transform(results[0],results[1],results[2],shape);
					hdcom.Save2DTifImage_RGB(output, shape, outpath+"/denoise_iterations/", filename+"_iter"+aux::zfill_int2string(iter,2));
				}
				if (params->noiselevel.continuous_estimate)
				{
					sigma1 = noise.get_noiselevel_2D_RGB(results[0], results[1], results[2], params);
					std::cout << "noise level " << iter << ": " << sigma1[0] << ", " << sigma1[1] << ", " << sigma1[2] << "          " << std::endl;
					if (params->color.average_channelsigma) sigma1[0] = sigma1[1] = sigma1[2] = (sigma1[0]+sigma1[1]+sigma1[2])/3.f;

					sigma0[0] = params->alpha*sigma0[0] + (1.f-params->alpha)*sigma1[0];
					sigma0[1] = params->alpha*sigma0[1] + (1.f-params->alpha)*sigma1[1];
					sigma0[2] = params->alpha*sigma0[2] + (1.f-params->alpha)*sigma1[2];
				}
			}
			/////////////////////////////////////////////////////////////////////

			//Create RGB output
			//////////////////////////////////////////////////////////////////
			if(params->color.stabilize_variance) color::inverse_anscombe_transform(results[0],results[1],results[2],shape);
			//color::HSV2RGB(results[0], results[1], results[2], shape);
			for (int idx = 0; idx < nslice; idx++)
			{
				output[3*idx] = (uint8_t) std::min(255.f, results[0][idx]);
				output[3*idx+1] = (uint8_t) std::min(255.f, results[1][idx]);
				output[3*idx+2] = (uint8_t) std::min(255.f, results[2][idx]);
			}
			hdcom.Save2DTifImage_RGB(output, shape, outpath, filename+"_denoised");
			//////////////////////////////////////////////////////////////////

			free(image_raw[0]); free(image_raw[1]); free(image_raw[2]);
			free(results[0]); free(results[1]); free(results[2]);
			free(output);
		}
		std::cout << "--------------------------------------" << std::endl;

		free(sigmalist);

		return;
	}
}

#endif //COLOR_DENOISING_H

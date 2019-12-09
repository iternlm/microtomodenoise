#include "noiselevel.h"
#include <math.h>
#include <algorithm>
#include "../Geometry/hdcommunication.h"
#include "../Geometry/auxiliary.h"

namespace noise
{
	float* NoiseLevel::get_noiselevel(float *imagestack, std::vector<std::string> &filelist, protocol::DenoiseParameters *params)
	{
		float *sigma = (float*) calloc(datashape[2], sizeof(*sigma));
		int blocksize = params->cpu.max_threads;

		if (params->noiselevel.mode == "z-adaptive" || params->noiselevel.mode == "auto" || params->noiselevel.mode == "global" || params->noiselevel.mode == "semimanual")
		{
			std::vector<std::vector<float>> noiseproperties;
			int circular_mask_diameter = params->noiselevel.circular_mask_diameter;

			bool firstevaluation = true;
			if (samples.size() == 0) samples = drawsamples(circular_mask_diameter);
			else firstevaluation = false;

			float *patchstd;
			if (!params->cpu.blockwise || (params->gpu.n_gpus > 0 && !params->gpu.blockwise_host)) patchstd = get_patchstd(samples, imagestack);
			else if (firstevaluation || params->io.firstslice == 0) patchstd = get_patchstd(samples, filelist, blocksize, params->io.firstslice);
			else {
				std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";
				int offset = 0;

				for (unsigned int i = 0; i < filelist.size(); i++)
					if (filelist[i].substr(filelist[i].length()-8,8) == fs_sub){offset = i; break;}

				patchstd = get_patchstd(samples, filelist, blocksize, offset);
			}

			if (params->noiselevel.mode == "z-adaptive")
			{
				noiseproperties = create2clusters_zwindow(samples, patchstd, params->noiselevel.zwindow);

				if ((int) noiseproperties.size() != datashape[2])
					std::cout << "Warning! Insufficient noise sampling!" << std::endl;

				for (int i = 0; i < datashape[2]; i++)
					sigma[(int) noiseproperties[i][0]] = noiseproperties[i][1] + params->noiselevel.stds_from_mean*noiseproperties[i][2];
			}
			else if (params->noiselevel.mode == "auto" || params->noiselevel.mode == "global")
			{
				std::pair<float,float> global_noise = create2clusters(patchstd, samples.size()); //mean and std of the noise level

				for (int i = 0; i < datashape[2]; i++)
					sigma[i] = global_noise.first + params->noiselevel.stds_from_mean*global_noise.second;
			}
			else if (params->noiselevel.mode == "semimanual")
			{
				if (firstevaluation)
				{
					float smallest_distance = 1e9;
					float sigma_val = params->noiselevel.sigma[0];

					for (unsigned int i = 0; i < samples.size(); i++)
					{
						if(fabs(patchstd[i]-sigma_val) < smallest_distance)
						{
							smallest_distance = fabs(patchstd[i]-sigma_val);
							reference_sample_semimanual = i;
						}
					}

					for (int i = 0; i < datashape[2]; i++) sigma[i] = sigma_val;
				}
				else
				{
					float sigma_val = patchstd[reference_sample_semimanual];
					params->noiselevel.sigma[1] = sigma_val;
					for (int i = 0; i < datashape[2]; i++) sigma[i] = sigma_val;
				}
			}

			free(patchstd);
		}
		else if (params->noiselevel.mode == "manual")
		{
			int noisepos = 0;
			if (ncalls > 0) noisepos = 1;

			for (int i = 0; i < datashape[2]; i++)
				sigma[i] = params->noiselevel.sigma[noisepos];
		}
		else
			std::cout << "Error! Unknown noise estimation mode!" << std::endl;

		ncalls++;

		return sigma;
	}
	std::vector<float> NoiseLevel::get_noiselevel_2D_RGB(float *R, float *G, float *B, protocol::DenoiseParameters *params)
	{
		float *sigma = (float*) calloc(datashape[2], sizeof(*sigma));

		std::vector<float> RGBsigma(3, 0.0);

		if (params->noiselevel.mode == "z-adaptive" || params->noiselevel.mode == "auto" || params->noiselevel.mode == "global" || params->noiselevel.mode == "semimanual")
		{
			std::vector<std::vector<float>> noiseproperties;
			int circular_mask_diameter = params->noiselevel.circular_mask_diameter;

			bool firstevaluation = true;
			if (samples.size() == 0) samples = drawsamples(circular_mask_diameter);
			else firstevaluation = false;

			float *patchstd_R = get_patchstd(samples, R);
			float *patchstd_G = get_patchstd(samples, G);
			float *patchstd_B = get_patchstd(samples, B);

			if (params->noiselevel.mode == "z-adaptive" || params->noiselevel.mode == "auto" || params->noiselevel.mode == "global")
			{
				std::pair<float,float> global_noise_R = create2clusters(patchstd_R, samples.size());
				std::pair<float,float> global_noise_G = create2clusters(patchstd_G, samples.size());
				std::pair<float,float> global_noise_B = create2clusters(patchstd_B, samples.size());

				RGBsigma[0] = global_noise_R.first + params->noiselevel.stds_from_mean*global_noise_R.second;
				RGBsigma[1] = global_noise_G.first + params->noiselevel.stds_from_mean*global_noise_G.second;
				RGBsigma[2] = global_noise_B.first + params->noiselevel.stds_from_mean*global_noise_B.second;
			}
			else if (params->noiselevel.mode == "semimanual")
			{
				if (firstevaluation)
				{
					float smallest_distance_R = 1e9;
					float smallest_distance_G = 1e9;
					float smallest_distance_B = 1e9;

					float sigma_val = params->noiselevel.sigma[0];

					for (unsigned int i = 0; i < samples.size(); i++)
					{
						if(fabs(patchstd_R[i]-sigma_val) < smallest_distance_R)
						{
							smallest_distance_R = fabs(patchstd_R[i]-sigma_val);
							reference_sample_semimanual = i;
						}
						if(fabs(patchstd_G[i]-sigma_val) < smallest_distance_G)
						{
							smallest_distance_G = fabs(patchstd_G[i]-sigma_val);
							reference_sample_semimanualG = i;
						}
						if(fabs(patchstd_B[i]-sigma_val) < smallest_distance_B)
						{
							smallest_distance_B = fabs(patchstd_B[i]-sigma_val);
							reference_sample_semimanualB = i;
						}
					}

					RGBsigma[0] = sigma_val;
					RGBsigma[1] = sigma_val;
					RGBsigma[2] = sigma_val;
				}
				else
				{
					float sigma_val = patchstd_R[reference_sample_semimanual];
					float sigma_valG = patchstd_G[reference_sample_semimanualG];
					float sigma_valB = patchstd_B[reference_sample_semimanualB];
					params->noiselevel.sigma[1] = sigma_val;

					RGBsigma[0] = sigma_val;
					RGBsigma[1] = sigma_valG;
					RGBsigma[2] = sigma_valB;
				}
			}

			free(patchstd_R);
			free(patchstd_G);
			free(patchstd_B);
		}
		else if (params->noiselevel.mode == "manual")
		{
			int noisepos = 0;
			if (ncalls > 0) noisepos = 1;

			RGBsigma[0] = params->noiselevel.sigma[noisepos];
			RGBsigma[1] = params->noiselevel.sigma[noisepos];
			RGBsigma[2]  = params->noiselevel.sigma[noisepos];
		}
		else
			std::cout << "Error! Unknown noise estimation mode!" << std::endl;

		ncalls++;

		return RGBsigma;
	}

	std::vector<long long int> NoiseLevel::drawsamples(int circular_mask_diameter)
    {
		//select positions in the imagestack for sampling the noise standard deviation

        int nx = datashape[0];
        int ny = datashape[1];
        int nz = datashape[2];
        long long int nslice = nx*ny;

        std::vector<long long int> idx_list(n_samples, -1);

        float maxradiussq = circular_mask_diameter/2.f-sqrt(2*patchsize*patchsize);
        maxradiussq *= maxradiussq;

		#pragma omp parallel for
        for (uint64_t sample = 0; sample < n_samples; sample++)
        {
            int x = (rand() % (nx-patchsize));
            int y = (rand() % (ny-patchsize));
            int z = (rand() % nz);

            if (circular_mask_diameter > 0)
            {
                float radiussq = (x-nx/2)*(x-nx/2)+(y-ny/2)*(y-ny/2);

                while (radiussq > maxradiussq)
                {
                    x = (rand() % (nx-patchsize));
                    y = (rand() % (ny-patchsize));

                    radiussq = (x-nx/2)*(x-nx/2)+(y-ny/2)*(y-ny/2);
                }
            }

            long long int this_idx = z*nslice + y*nx + x;

            idx_list[sample] = this_idx;
        }

        std::sort(idx_list.begin(), idx_list.end());

        return idx_list;
    }
	float* NoiseLevel::get_patchstd(std::vector<long long int> &samples, float *imagestack)
	{
		//extract the local standard deviation from 2D patches

		int nx = datashape[0];
		long long int nslice = datashape[0]*datashape[1];
		int patchlength = patchsize*patchsize;

		float* patchstd = (float*) calloc(samples.size(), sizeof(*patchstd));

		#pragma omp parallel for
		for (unsigned long long int sample = 0; sample < samples.size(); sample++)
		{
			long long int idx = samples[sample];
			int z0 = idx/nslice;
			int y0 = (idx-z0*nslice)/nx;
			int x0 = idx-z0*nslice-y0*nx;

			float sum = 0;

			for (int y = y0; y < y0+patchsize; y++)
			{
				long long int pos0 = z0*nslice + y*nx + x0;
				for (int x = 0; x < patchsize; x++)
					sum += imagestack[pos0+x];
			}

			float mean = sum / patchlength;

			float sq_sum = 0.0f;

			for (int y = y0; y < y0+patchsize; y++)
			{
				long long int pos0 = z0*nslice + y*nx + x0;

				for (int x = 0; x < patchsize; x++)
				{
					float val = imagestack[pos0+x];
					sq_sum += (val-mean)*(val-mean);
				}
			}

			float variance = sq_sum / patchlength;

			patchstd[sample] = sqrtf(variance);
		}

		return patchstd;
	}
	float* NoiseLevel::get_patchstd(std::vector<long long int> &samples, std::vector<std::string> &filelist, int blocksize, int zoffset)
	{
		//extract the local standard deviation from 2D patches

		int nx = datashape[0];
		long long int nslice = datashape[0]*datashape[1];
		int patchlength = patchsize*patchsize;

		int n_blocks = datashape[2]/blocksize;
		if ((datashape[2]%blocksize) != 0) n_blocks++;

		float* patchstd = (float*) calloc(samples.size(), sizeof(*patchstd));

		int firstsample = 0;
		int lastsample = 0;

		hdcom::HdCommunication hdcom;
		int shape[3];

		for (int n = 0; n < n_blocks; n++)
		{
			int firstslice = n*blocksize;
			int lastslice = std::min((n+1)*blocksize-1, datashape[2]-1);

			float* imageblock = hdcom.Get3DTifSequence_32bitPointer(filelist, shape, firstslice+zoffset, lastslice+zoffset);

			//select samples that are in the substack
			for (unsigned long long int sample = firstsample; sample < samples.size(); sample++)
			{
				long long int idx = samples[sample];
				int z0 = idx/nslice;

				if (z0 > lastslice){lastsample = sample-1; break;}
				lastsample = sample;
			}

			#pragma omp parallel for
			for (long long int sample = firstsample; sample <= lastsample; sample++)
			{
				long long int idx = samples[sample];
				int z0 = idx/nslice;
				int y0 = (idx-z0*nslice)/nx;
				int x0 = idx-z0*nslice-y0*nx;

				z0 -= firstslice;

				float sum = 0;

				for (int y = y0; y < y0+patchsize; y++)
				{
					long long int pos0 = z0*nslice + y*nx + x0;
					for (int x = 0; x < patchsize; x++)
						sum += imageblock[pos0+x];
				}

				float mean = sum / patchlength;

				float sq_sum = 0.0f;

				for (int y = y0; y < y0+patchsize; y++)
				{
					long long int pos0 = z0*nslice + y*nx + x0;

					for (int x = 0; x < patchsize; x++)
					{
						float val = imageblock[pos0+x];
						sq_sum += (val-mean)*(val-mean);
					}
				}

				float variance = sq_sum / patchlength;

				patchstd[sample] = sqrtf(variance);
			}

			firstsample = lastsample+1;
			free(imageblock);
		}

		return patchstd;
	}

	std::pair<float,float> NoiseLevel::create2clusters(float *values, int n_samples, int firstpos, int lastpos)
	{
		/*
		 * returns the smaller of two cluster centers as the best noise level along with its standard deviation.
		 * zero-values are excluded.
		 */

		int fs = 0;
		int ls = n_samples-1;

		if (firstpos != -1) fs = firstpos;
		if (lastpos != -1) ls = lastpos;

		float minimum = 1e20f;
		float maximum = -1e20f;

		//get max/min and initialize cluster means
		////////////////////////////////////////////////
		//#pragma omp parallel for reduction(max: maximum), reduction(min: minimum)
		for (int idx = fs; idx <= ls; idx++)
		{
			float val = values[idx];
			if (val < minimum) minimum = val;
			if (val > maximum) maximum = val;
		}
		float mean1 = minimum + 0.25*(maximum-minimum);
		float mean2 = maximum - 0.25*(maximum-minimum);
		////////////////////////////////////////////////

		//initial assignment excluding zero-variance
		////////////////////////////////////////////////
		int* assignment = (int*) malloc(n_samples*sizeof(*assignment));

		float cumsum1 = 0.0f;
		float cumsum2 = 0.0f;
		float sum1 = 0.0f;
		float sum2 = 0.0f;

		//#pragma omp parallel for reduction(+: cumsum1, cumsum2, sum1, sum2)
		for (int idx = fs; idx <= ls; idx++)
		{
			float val = values[idx];

			if (val == 0.0f) assignment[idx] = -1;
			else if (fabs(val-mean1) < fabs(val-mean2))
			{
				cumsum1 += val;
				sum1++;
				assignment[idx] = 0;
			}
			else
			{
				cumsum2 += val;
				sum2++;
				assignment[idx] = 1;
			}
		}

		if(sum1 > 0.0f) mean1 = cumsum1/sum1;
		if(sum2 > 0.0f) mean2 = cumsum2/sum2;

		if (sum1 == 0 && sum2 == 0)
			return {0.0, 0.0};
		////////////////////////////////////////////////

		//cluster variance
		////////////////////////////////////////////////
		float var1 = 0.0f;
		float var2 = 0.0f;

		//#pragma omp parallel for reduction(+: var1, var2)
		for (int idx = fs; idx <= ls; idx++)
		{
			float val = values[idx];
			float assigned = assignment[idx];

			if (assigned == 0)
				var1 += (val-mean1)*(val-mean1);
			else if (assigned == 1)
				var2 += (val-mean2)*(val-mean2);
		}

		var1 /= std::max(1.0f,sum1);
		var2 /= std::max(1.0f,sum2);
		////////////////////////////////////////////////

		//maximize assignment probability
		////////////////////////////////////////////////
		float PI = 3.141592654f;
		bool change = true;

		uint64_t iters_taken = 0;
		uint64_t maxiters = 1000;

		while (change && var1 > 0.0f && var2 > 0.0f)
		{
			change = false;
			iters_taken++;

			cumsum1 = cumsum2 = sum1 = sum2 = 0.0f;

			//#pragma omp parallel for reduction(+: cumsum1, cumsum2, sum1, sum2)
			for (int idx = fs; idx <= ls; idx++)
			{
				float val = values[idx];
				if (val == 0.0f) continue;

				float prob1 = 1./(sqrtf(var1*2*PI))*expf(-((val-mean1)*(values[idx]-mean1)/(2*var1)));
				float prob2 = 1./(sqrtf(var2*2*PI))*expf(-((val-mean2)*(values[idx]-mean2)/(2*var2)));

				if (prob2 > prob1)
				{
					if (assignment[idx] != 1)
					{
						change = true;
						assignment[idx] = 1;
					}
					cumsum2 += val;
					sum2++;
				}
				else
				{
					if (assignment[idx] != 0)
					{
						change = true;
						assignment[idx] = 0;
					}
					cumsum1 += val;
					sum1++;
				}
			}

			if(sum1 > 0.0f) mean1 = cumsum1/sum1;
			if(sum2 > 0.0f) mean2 = cumsum2/sum2;
			var1 = 0.;
			var2 = 0.;

			var1 = 0.0f;
			var2 = 0.0f;

			//#pragma omp parallel for reduction(+: var1, var2)
			for (int idx = fs; idx <= ls; idx++)
			{
				float val = values[idx];
				float assigned = assignment[idx];

				if (assigned == 0)
					var1 += (val-mean1)*(val-mean1);
				else if (assigned == 1)
					var2 += (val-mean2)*(val-mean2);
			}

			var1 /= std::max(1.0f,sum1);
			var2 /= std::max(1.0f,sum2);

			if (iters_taken == maxiters) break;
		}

		free(assignment);

		if (mean1 <= mean2 || var2 <= 0.0) return {mean1, sqrt(var1)};
		return {mean2, sqrt(var2)};
	}
	std::vector<std::vector<float>> NoiseLevel::create2clusters_zwindow(std::vector<long long int> &samples, float *values, int windowsize)
	{
		long long int nslice = datashape[0]*datashape[1];
		int lastpos = datashape[2];
		int startpos = 0;
		int endpos = 0;

		float max_std = -1.f;
		bool zero_variance_slice = false;

		int pos = 0;
		std::vector<std::vector<int>> windows;

		while(pos < lastpos)
		{
			while (samples[startpos]/nslice <= (pos-windowsize/2))
				startpos++;
			while ((endpos < (int) (samples.size()-1)) && (samples[endpos]/nslice <= (pos+windowsize/2)))
				endpos++;

			windows.push_back({pos, startpos, endpos});
			pos++;
		}

		std::vector<std::vector<float>> output(windows.size()); //list with zpos, noise std, std of noise std

		#pragma omp parallel for reduction(max: max_std)
		for (unsigned int i = 0; i < windows.size(); i++)
		{
			int this_pos = windows[i][0];
			int this_startpos = windows[i][1];
			int this_endpos = windows[i][2];

			std::pair<float,float> this_params = create2clusters(values, samples.size(), this_startpos, this_endpos);
			std::vector<float> this_output = {(float) this_pos, this_params.first, this_params.second};

			if (this_params.first > max_std) max_std = this_params.first;
			if (this_params.first == 0.0f) zero_variance_slice = true; //no valid sample in slice;

			output[i] = this_output;
		}

		//If there are missing slices, extrapolate!
		if (zero_variance_slice && max_std > 0.0f)
		{
			while (zero_variance_slice)
			{
				zero_variance_slice = false;

				for (int i = 0; i < (int) output.size(); i++)
				{
					float this_var = output[i][1];

					if (this_var == 0.0f)
					{
						float new_var = 0.0f;
						float new_varstd = 0.0f;
						int count = 0;

						if (i > 0 && output[i-1][1] > 0.0f) {new_var += output[i-1][1]; new_varstd += output[i-1][2]; count++;}
						if (i < (int) (output.size()-1)
								&& output[i+1][1] > 0.0f) {new_var += output[i+1][1]; new_varstd += output[i+1][2]; count++;}

						if (count > 0){
							this_var = new_var/count;
							output[i][1] = this_var;
							output[i][2] = new_varstd/count;
						}
					}

					if (this_var == 0.0f)
						zero_variance_slice = true;
				}
			}
		}

		return output;
	}

}

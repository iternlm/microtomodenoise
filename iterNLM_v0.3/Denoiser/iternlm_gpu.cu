#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <chrono>
#include <fstream>

#include "iternlm_gpu.h"
#include "gpu_constants.cuh"
#include "../Geometry/hdcommunication.h"
#include "../Geometry/auxiliary.h"
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
	namespace gpu_denoise
	{
		__device__ __inline__ float expapproximation(float x){return (120.f + 60.f*x + 12.f*x*x + x*x*x)/(120.f - 60.f*x + 12.f*x*x - x*x*x);} //3rd order Pade approximation

		__global__ void pad_reflective(float *input, float *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx0 = gpu_const::nx;
			int ny0 = gpu_const::ny;
			int nz0 = gpu_const::nz;

			idx_type nslice0 = nx0*ny0;

			int padding0 = gpu_const::padding[0];
			int padding1 = gpu_const::padding[1];
			int padding2 = gpu_const::padding[2];

			int nx1 = nx0 + padding0 + gpu_const::padding[3];
			int ny1 = ny0 + padding1 + gpu_const::padding[4];
			int nz1 = nz0 + padding2 + gpu_const::padding[5];

			idx_type nslice1 = nx1*ny1;
			idx_type nstack1 = nz1*nslice1;

			idx_type idx1 = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx1 >= nstack1) idx1 = threadIdx.x;

			int z1 = idx1/nslice1;
			int y1 = (idx1-z1*nslice1)/nx1;
			int x1 = idx1-z1*nslice1-y1*nx1;

			int z0 = z1-padding2;
			int y0 = y1-padding1;
			int x0 = x1-padding0;

			while (z0 < 0 || z0 >= nz0 || y0 < 0 || y0 >= ny0 || x0 < 0 || x0 >= nx0)
			{
				if (z0 < 0) z0 = -z0;
				if (y0 < 0) y0 = -y0;
				if (x0 < 0) x0 = -x0;

				if (z0 >= nz0) z0 = 2*nz0-z0-2;
				if (y0 >= ny0) y0 = 2*ny0-y0-2;
				if (x0 >= nx0) x0 = 2*nx0-x0-2;
			}

			long long int idx0 = z0*nslice0+y0*nx0+x0;

			__syncthreads();
			/////////////////////////////////////////////

			output[idx1] = input[idx0];

			return;
		}

		__global__ void apply_filter_generic(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions, float *distweight)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			int nsize_patch = gpu_const::nsize_patch;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];
			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;
				float tmp = image_previous[idx1]-image_previous[idx0];
				distance += (tmp*tmp)*distweight[0];

				#pragma unroll
				for (int p = 1; p < nsize_patch; p++)
				{
					tmp = image_previous[idx1 + patch_positions[p]]-image_previous[idx0 + patch_positions[p]];
					distance += (tmp*tmp)*distweight[p];
				}
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch111(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1 =  patch_positions[1];
			idx_type ppos2 =  patch_positions[2];
			idx_type ppos3 =  patch_positions[3];
			idx_type ppos4 =  patch_positions[4];
			idx_type ppos5 =  patch_positions[5];
			idx_type ppos6 =  patch_positions[6];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0];
			float val_orig1 = image_previous[idx0 + ppos1];
			float val_orig2 = image_previous[idx0 + ppos2];
			float val_orig3 = image_previous[idx0 + ppos3];
			float val_orig4 = image_previous[idx0 + ppos4];
			float val_orig5 = image_previous[idx0 + ppos5];
			float val_orig6 = image_previous[idx0 + ppos6];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.142857143f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.142857143f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;

				//float this_weight = (distance > -3.56648f) ? expapproximation(distance) : 0.0f;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch112(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1= patch_positions[1];idx_type ppos2= patch_positions[2];idx_type ppos3= patch_positions[3];idx_type ppos4= patch_positions[4];
			idx_type ppos5= patch_positions[5];idx_type ppos6= patch_positions[6];idx_type ppos7= patch_positions[7];idx_type ppos8= patch_positions[8];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ];
			float val_orig1=image_previous[idx0+ppos1];float val_orig2=image_previous[idx0+ppos2];
			float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
			float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];float val_orig7=image_previous[idx0+ppos7];
			float val_orig8=image_previous[idx0+ppos8];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.04f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch113(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1= patch_positions[1];idx_type ppos2= patch_positions[2];idx_type ppos3= patch_positions[3];idx_type ppos4= patch_positions[4];
			idx_type ppos5= patch_positions[5];idx_type ppos6= patch_positions[6];idx_type ppos7= patch_positions[7];idx_type ppos8= patch_positions[8];
			idx_type ppos9= patch_positions[9];idx_type ppos10= patch_positions[10];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ];float val_orig1=image_previous[idx0+ppos1];
			float val_orig2=image_previous[idx0+ppos2];float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
			float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];
			float val_orig7=image_previous[idx0+ppos7];float val_orig8=image_previous[idx0+ppos8];
			float val_orig9=image_previous[idx0+ppos9];float val_orig10=image_previous[idx0+ppos10];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.0204082f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch221(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1 =  patch_positions[1];
			idx_type ppos2 =  patch_positions[2];
			idx_type ppos3 =  patch_positions[3];
			idx_type ppos4 =  patch_positions[4];
			idx_type ppos5 =  patch_positions[5];
			idx_type ppos6 =  patch_positions[6];
			idx_type ppos7 =  patch_positions[7];
			idx_type ppos8 =  patch_positions[8];
			idx_type ppos9 =  patch_positions[9];
			idx_type ppos10=  patch_positions[10];
			idx_type ppos11=  patch_positions[11];
			idx_type ppos12=  patch_positions[12];
			idx_type ppos13=  patch_positions[13];
			idx_type ppos14=  patch_positions[14];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ];
			float val_orig1 = image_previous[idx0 + ppos1];
			float val_orig2 = image_previous[idx0 + ppos2];
			float val_orig3 = image_previous[idx0 + ppos3];
			float val_orig4 = image_previous[idx0 + ppos4];
			float val_orig5 = image_previous[idx0 + ppos5];
			float val_orig6 = image_previous[idx0 + ppos6];
			float val_orig7 = image_previous[idx0 + ppos7];
			float val_orig8 = image_previous[idx0 + ppos8];
			float val_orig9 = image_previous[idx0 + ppos9];
			float val_orig10= image_previous[idx0 + ppos10];
			float val_orig11= image_previous[idx0 + ppos11];
			float val_orig12= image_previous[idx0 + ppos12];
			float val_orig13= image_previous[idx0 + ppos13];
			float val_orig14= image_previous[idx0 + ppos14];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos11]-val_orig11; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos12]-val_orig12; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos13]-val_orig13; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos14]-val_orig14; distance += (tmp*tmp)*0.111111f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch222(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1 =  patch_positions[1]; idx_type ppos2 =  patch_positions[2];
			idx_type ppos3 =  patch_positions[3]; idx_type ppos4 =  patch_positions[4];
			idx_type ppos5 =  patch_positions[5]; idx_type ppos6 =  patch_positions[6];
			idx_type ppos7 =  patch_positions[7]; idx_type ppos8 =  patch_positions[8];
			idx_type ppos9 =  patch_positions[9]; idx_type ppos10=  patch_positions[10];
			idx_type ppos11=  patch_positions[11];idx_type ppos12=  patch_positions[12];
			idx_type ppos13=  patch_positions[13];idx_type ppos14=  patch_positions[14];
			idx_type ppos15=  patch_positions[15];idx_type ppos16=  patch_positions[16];
			idx_type ppos17=  patch_positions[17];idx_type ppos18=  patch_positions[18];
			idx_type ppos19=  patch_positions[19];idx_type ppos20=  patch_positions[20];
			idx_type ppos21=  patch_positions[21];idx_type ppos22=  patch_positions[22];
			idx_type ppos23=  patch_positions[23];idx_type ppos24=  patch_positions[24];
			idx_type ppos25=  patch_positions[25];idx_type ppos26=  patch_positions[26];
			idx_type ppos27=  patch_positions[27];idx_type ppos28=  patch_positions[28];
			idx_type ppos29=  patch_positions[29];idx_type ppos30=  patch_positions[30];
			idx_type ppos31=  patch_positions[31];idx_type ppos32=  patch_positions[32];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ]; float val_orig1 = image_previous[idx0 + ppos1];
			float val_orig2 = image_previous[idx0 + ppos2]; float val_orig3 = image_previous[idx0 + ppos3];
			float val_orig4 = image_previous[idx0 + ppos4]; float val_orig5 = image_previous[idx0 + ppos5];
			float val_orig6 = image_previous[idx0 + ppos6]; float val_orig7 = image_previous[idx0 + ppos7];
			float val_orig8 = image_previous[idx0 + ppos8]; float val_orig9 = image_previous[idx0 + ppos9];
			float val_orig10= image_previous[idx0 + ppos10];float val_orig11= image_previous[idx0 + ppos11];
			float val_orig12= image_previous[idx0 + ppos12];float val_orig13= image_previous[idx0 + ppos13];
			float val_orig14= image_previous[idx0 + ppos14];float val_orig15= image_previous[idx0 + ppos15];
			float val_orig16= image_previous[idx0 + ppos16];float val_orig17= image_previous[idx0 + ppos17];
			float val_orig18= image_previous[idx0 + ppos18];float val_orig19= image_previous[idx0 + ppos19];
			float val_orig20= image_previous[idx0 + ppos20];float val_orig21= image_previous[idx0 + ppos21];
			float val_orig22= image_previous[idx0 + ppos22];float val_orig23= image_previous[idx0 + ppos23];
			float val_orig24= image_previous[idx0 + ppos24];float val_orig25= image_previous[idx0 + ppos25];
			float val_orig26= image_previous[idx0 + ppos26];float val_orig27= image_previous[idx0 + ppos27];
			float val_orig28= image_previous[idx0 + ppos28];float val_orig29= image_previous[idx0 + ppos29];
			float val_orig30= image_previous[idx0 + ppos30];float val_orig31= image_previous[idx0 + ppos31];
			float val_orig32= image_previous[idx0 + ppos32];


			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos11]-val_orig11; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos12]-val_orig12; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos13]-val_orig13; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos14]-val_orig14; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos15]-val_orig15; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos16]-val_orig16; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos17]-val_orig17; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos18]-val_orig18; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos19]-val_orig19; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos20]-val_orig20; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos21]-val_orig21; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos22]-val_orig22; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos23]-val_orig23; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos24]-val_orig24; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos25]-val_orig25; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos26]-val_orig26; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos27]-val_orig27; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos28]-val_orig28; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos29]-val_orig29; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos30]-val_orig30; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos31]-val_orig31; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos32]-val_orig32; distance += (tmp*tmp)*0.04f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch331(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1 =  patch_positions[1]; idx_type ppos2 =  patch_positions[2];
			idx_type ppos3 =  patch_positions[3]; idx_type ppos4 =  patch_positions[4];
			idx_type ppos5 =  patch_positions[5]; idx_type ppos6 =  patch_positions[6];
			idx_type ppos7 =  patch_positions[7]; idx_type ppos8 =  patch_positions[8];
			idx_type ppos9 =  patch_positions[9]; idx_type ppos10=  patch_positions[10];
			idx_type ppos11=  patch_positions[11];idx_type ppos12=  patch_positions[12];
			idx_type ppos13=  patch_positions[13];idx_type ppos14=  patch_positions[14];
			idx_type ppos15=  patch_positions[15];idx_type ppos16=  patch_positions[16];
			idx_type ppos17=  patch_positions[17];idx_type ppos18=  patch_positions[18];
			idx_type ppos19=  patch_positions[19];idx_type ppos20=  patch_positions[20];
			idx_type ppos21=  patch_positions[21];idx_type ppos22=  patch_positions[22];
			idx_type ppos23=  patch_positions[23];idx_type ppos24=  patch_positions[24];
			idx_type ppos25=  patch_positions[25];idx_type ppos26=  patch_positions[26];
			idx_type ppos27=  patch_positions[27];idx_type ppos28=  patch_positions[28];
			idx_type ppos29=  patch_positions[29];idx_type ppos30=  patch_positions[30];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ]; float val_orig1 = image_previous[idx0 + ppos1];
			float val_orig2 = image_previous[idx0 + ppos2]; float val_orig3 = image_previous[idx0 + ppos3];
			float val_orig4 = image_previous[idx0 + ppos4]; float val_orig5 = image_previous[idx0 + ppos5];
			float val_orig6 = image_previous[idx0 + ppos6]; float val_orig7 = image_previous[idx0 + ppos7];
			float val_orig8 = image_previous[idx0 + ppos8]; float val_orig9 = image_previous[idx0 + ppos9];
			float val_orig10= image_previous[idx0 + ppos10];float val_orig11= image_previous[idx0 + ppos11];
			float val_orig12= image_previous[idx0 + ppos12];float val_orig13= image_previous[idx0 + ppos13];
			float val_orig14= image_previous[idx0 + ppos14];float val_orig15= image_previous[idx0 + ppos15];
			float val_orig16= image_previous[idx0 + ppos16];float val_orig17= image_previous[idx0 + ppos17];
			float val_orig18= image_previous[idx0 + ppos18];float val_orig19= image_previous[idx0 + ppos19];
			float val_orig20= image_previous[idx0 + ppos20];float val_orig21= image_previous[idx0 + ppos21];
			float val_orig22= image_previous[idx0 + ppos22];float val_orig23= image_previous[idx0 + ppos23];
			float val_orig24= image_previous[idx0 + ppos24];float val_orig25= image_previous[idx0 + ppos25];
			float val_orig26= image_previous[idx0 + ppos26];float val_orig27= image_previous[idx0 + ppos27];
			float val_orig28= image_previous[idx0 + ppos28];float val_orig29= image_previous[idx0 + ppos29];
			float val_orig30= image_previous[idx0 + ppos30];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos11]-val_orig11; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos12]-val_orig12; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos13]-val_orig13; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos14]-val_orig14; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos15]-val_orig15; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos16]-val_orig16; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos17]-val_orig17; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos18]-val_orig18; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos19]-val_orig19; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos20]-val_orig20; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos21]-val_orig21; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos22]-val_orig22; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos23]-val_orig23; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos24]-val_orig24; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos25]-val_orig25; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos26]-val_orig26; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos27]-val_orig27; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos28]-val_orig28; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos29]-val_orig29; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos30]-val_orig30; distance += (tmp*tmp)*0.111111f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch332(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			idx_type ppos1= patch_positions[1];idx_type ppos2= patch_positions[2];idx_type ppos3= patch_positions[3];idx_type ppos4= patch_positions[4];
			idx_type ppos5= patch_positions[5];idx_type ppos6= patch_positions[6];idx_type ppos7= patch_positions[7];idx_type ppos8= patch_positions[8];
			idx_type ppos9= patch_positions[9];idx_type ppos10= patch_positions[10];idx_type ppos11= patch_positions[11];idx_type ppos12= patch_positions[12];
			idx_type ppos13= patch_positions[13];idx_type ppos14= patch_positions[14];idx_type ppos15= patch_positions[15];idx_type ppos16= patch_positions[16];
			idx_type ppos17= patch_positions[17];idx_type ppos18= patch_positions[18];idx_type ppos19= patch_positions[19];idx_type ppos20= patch_positions[20];
			idx_type ppos21= patch_positions[21];idx_type ppos22= patch_positions[22];idx_type ppos23= patch_positions[23];idx_type ppos24= patch_positions[24];
			idx_type ppos25= patch_positions[25];idx_type ppos26= patch_positions[26];idx_type ppos27= patch_positions[27];idx_type ppos28= patch_positions[28];
			idx_type ppos29= patch_positions[29];idx_type ppos30= patch_positions[30];idx_type ppos31= patch_positions[31];idx_type ppos32= patch_positions[32];
			idx_type ppos33= patch_positions[33];idx_type ppos34= patch_positions[34];idx_type ppos35= patch_positions[35];idx_type ppos36= patch_positions[36];
			idx_type ppos37= patch_positions[37];idx_type ppos38= patch_positions[38];idx_type ppos39= patch_positions[39];idx_type ppos40= patch_positions[40];
			idx_type ppos41= patch_positions[41];idx_type ppos42= patch_positions[42];idx_type ppos43= patch_positions[43];idx_type ppos44= patch_positions[44];
			idx_type ppos45= patch_positions[45];idx_type ppos46= patch_positions[46];idx_type ppos47= patch_positions[47];idx_type ppos48= patch_positions[48];
			idx_type ppos49= patch_positions[49];idx_type ppos50= patch_positions[50];idx_type ppos51= patch_positions[51];idx_type ppos52= patch_positions[52];
			idx_type ppos53= patch_positions[53];idx_type ppos54= patch_positions[54];idx_type ppos55= patch_positions[55];idx_type ppos56= patch_positions[56];
			idx_type ppos57= patch_positions[57];idx_type ppos58= patch_positions[58];idx_type ppos59= patch_positions[59];idx_type ppos60= patch_positions[60];
			idx_type ppos61= patch_positions[61];idx_type ppos62= patch_positions[62];idx_type ppos63= patch_positions[63];idx_type ppos64= patch_positions[64];
			idx_type ppos65= patch_positions[65];idx_type ppos66= patch_positions[66];idx_type ppos67= patch_positions[67];idx_type ppos68= patch_positions[68];
			idx_type ppos69= patch_positions[69];idx_type ppos70= patch_positions[70];idx_type ppos71= patch_positions[71];idx_type ppos72= patch_positions[72];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ];float val_orig1=image_previous[idx0+ppos1];float val_orig2=image_previous[idx0+ppos2];float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
			float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];float val_orig7=image_previous[idx0+ppos7];float val_orig8=image_previous[idx0+ppos8];
			float val_orig9=image_previous[idx0+ppos9];float val_orig10=image_previous[idx0+ppos10];float val_orig11=image_previous[idx0+ppos11];float val_orig12=image_previous[idx0+ppos12];
			float val_orig13=image_previous[idx0+ppos13];float val_orig14=image_previous[idx0+ppos14];float val_orig15=image_previous[idx0+ppos15];float val_orig16=image_previous[idx0+ppos16];
			float val_orig17=image_previous[idx0+ppos17];float val_orig18=image_previous[idx0+ppos18];float val_orig19=image_previous[idx0+ppos19];float val_orig20=image_previous[idx0+ppos20];
			float val_orig21=image_previous[idx0+ppos21];float val_orig22=image_previous[idx0+ppos22];float val_orig23=image_previous[idx0+ppos23];float val_orig24=image_previous[idx0+ppos24];
			float val_orig25=image_previous[idx0+ppos25];float val_orig26=image_previous[idx0+ppos26];float val_orig27=image_previous[idx0+ppos27];float val_orig28=image_previous[idx0+ppos28];
			float val_orig29=image_previous[idx0+ppos29];float val_orig30=image_previous[idx0+ppos30];float val_orig31=image_previous[idx0+ppos31];float val_orig32=image_previous[idx0+ppos32];
			float val_orig33=image_previous[idx0+ppos33];float val_orig34=image_previous[idx0+ppos34];float val_orig35=image_previous[idx0+ppos35];float val_orig36=image_previous[idx0+ppos36];
			float val_orig37=image_previous[idx0+ppos37];float val_orig38=image_previous[idx0+ppos38];float val_orig39=image_previous[idx0+ppos39];float val_orig40=image_previous[idx0+ppos40];
			float val_orig41=image_previous[idx0+ppos41];float val_orig42=image_previous[idx0+ppos42];float val_orig43=image_previous[idx0+ppos43];float val_orig44=image_previous[idx0+ppos44];
			float val_orig45=image_previous[idx0+ppos45];float val_orig46=image_previous[idx0+ppos46];float val_orig47=image_previous[idx0+ppos47];float val_orig48=image_previous[idx0+ppos48];
			float val_orig49=image_previous[idx0+ppos49];float val_orig50=image_previous[idx0+ppos50];float val_orig51=image_previous[idx0+ppos51];float val_orig52=image_previous[idx0+ppos52];
			float val_orig53=image_previous[idx0+ppos53];float val_orig54=image_previous[idx0+ppos54];float val_orig55=image_previous[idx0+ppos55];float val_orig56=image_previous[idx0+ppos56];
			float val_orig57=image_previous[idx0+ppos57];float val_orig58=image_previous[idx0+ppos58];float val_orig59=image_previous[idx0+ppos59];float val_orig60=image_previous[idx0+ppos60];
			float val_orig61=image_previous[idx0+ppos61];float val_orig62=image_previous[idx0+ppos62];float val_orig63=image_previous[idx0+ppos63];float val_orig64=image_previous[idx0+ppos64];
			float val_orig65=image_previous[idx0+ppos65];float val_orig66=image_previous[idx0+ppos66];float val_orig67=image_previous[idx0+ppos67];float val_orig68=image_previous[idx0+ppos68];
			float val_orig69=image_previous[idx0+ppos69];float val_orig70=image_previous[idx0+ppos70];float val_orig71=image_previous[idx0+ppos71];float val_orig72=image_previous[idx0+ppos72];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos11]-val_orig11; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos12]-val_orig12; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos13]-val_orig13; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos14]-val_orig14; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos15]-val_orig15; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos16]-val_orig16; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos17]-val_orig17; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos18]-val_orig18; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos19]-val_orig19; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos20]-val_orig20; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos21]-val_orig21; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos22]-val_orig22; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos23]-val_orig23; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos24]-val_orig24; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos25]-val_orig25; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos26]-val_orig26; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos27]-val_orig27; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos28]-val_orig28; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos29]-val_orig29; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos30]-val_orig30; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos31]-val_orig31; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos32]-val_orig32; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos33]-val_orig33; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos34]-val_orig34; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos35]-val_orig35; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos36]-val_orig36; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos37]-val_orig37; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos38]-val_orig38; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos39]-val_orig39; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos40]-val_orig40; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos41]-val_orig41; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos42]-val_orig42; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos43]-val_orig43; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos44]-val_orig44; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos45]-val_orig45; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos46]-val_orig46; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos47]-val_orig47; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos48]-val_orig48; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos49]-val_orig49; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos50]-val_orig50; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos51]-val_orig51; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos52]-val_orig52; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos53]-val_orig53; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos54]-val_orig54; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos55]-val_orig55; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos56]-val_orig56; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos57]-val_orig57; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos58]-val_orig58; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos59]-val_orig59; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos60]-val_orig60; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos61]-val_orig61; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos62]-val_orig62; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos63]-val_orig63; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos64]-val_orig64; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos65]-val_orig65; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos66]-val_orig66; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos67]-val_orig67; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos68]-val_orig68; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos69]-val_orig69; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos70]-val_orig70; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos71]-val_orig71; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos72]-val_orig72; distance += (tmp*tmp)*0.04f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
		__global__ void apply_filter_patch333(float *image_raw, float *image_previous, float *next_result, float *sigma_list, idx_type *search_positions, idx_type *patch_positions)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx;
			int ny = gpu_const::ny;
			int nz = gpu_const::nz;

			int xpad = gpu_const::padding[0];
			int ypad = gpu_const::padding[1];
			int zpad = gpu_const::padding[2];

			int nsize_search = gpu_const::nsize_search;
			float beta = gpu_const::beta;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			int nx_padded = nx+2*xpad;
			idx_type nslice_padded = nx_padded*(ny+2*ypad);

			idx_type idx_unpadded = (blockIdx.x*blockDim.x+threadIdx.x); //idx without padding
			if (idx_unpadded >= nstack) {idx_unpadded = threadIdx.x;}

			int z0 = idx_unpadded/nslice;
			int y0 = (idx_unpadded-z0*nslice)/nx;
			int x0 = idx_unpadded-z0*nslice-y0*nx;

			z0 += zpad;
			y0 += ypad;
			x0 += xpad;

			idx_type idx0 = z0*nslice_padded + y0*nx_padded + x0; //idx_padded

			float filtervalue = 0.0f;
			float filterweight = 0.0f;
			float maxweight = 0.0f;

			//float values_origin[115]; static memory is the second best choice
			/////////////////////////////////////////////

			__syncthreads();
			float noisy_value_origin = image_raw[idx0];

			//Putting this into constant memory doesn't help much
			idx_type ppos1= patch_positions[1];idx_type ppos2= patch_positions[2];idx_type ppos3= patch_positions[3];idx_type ppos4= patch_positions[4];
			idx_type ppos5= patch_positions[5];idx_type ppos6= patch_positions[6];idx_type ppos7= patch_positions[7];idx_type ppos8= patch_positions[8];
			idx_type ppos9= patch_positions[9];idx_type ppos10= patch_positions[10];idx_type ppos11= patch_positions[11];idx_type ppos12= patch_positions[12];
			idx_type ppos13= patch_positions[13];idx_type ppos14= patch_positions[14];idx_type ppos15= patch_positions[15];idx_type ppos16= patch_positions[16];
			idx_type ppos17= patch_positions[17];idx_type ppos18= patch_positions[18];idx_type ppos19= patch_positions[19];idx_type ppos20= patch_positions[20];
			idx_type ppos21= patch_positions[21];idx_type ppos22= patch_positions[22];idx_type ppos23= patch_positions[23];idx_type ppos24= patch_positions[24];
			idx_type ppos25= patch_positions[25];idx_type ppos26= patch_positions[26];idx_type ppos27= patch_positions[27];idx_type ppos28= patch_positions[28];
			idx_type ppos29= patch_positions[29];idx_type ppos30= patch_positions[30];idx_type ppos31= patch_positions[31];idx_type ppos32= patch_positions[32];
			idx_type ppos33= patch_positions[33];idx_type ppos34= patch_positions[34];idx_type ppos35= patch_positions[35];idx_type ppos36= patch_positions[36];
			idx_type ppos37= patch_positions[37];idx_type ppos38= patch_positions[38];idx_type ppos39= patch_positions[39];idx_type ppos40= patch_positions[40];
			idx_type ppos41= patch_positions[41];idx_type ppos42= patch_positions[42];idx_type ppos43= patch_positions[43];idx_type ppos44= patch_positions[44];
			idx_type ppos45= patch_positions[45];idx_type ppos46= patch_positions[46];idx_type ppos47= patch_positions[47];idx_type ppos48= patch_positions[48];
			idx_type ppos49= patch_positions[49];idx_type ppos50= patch_positions[50];idx_type ppos51= patch_positions[51];idx_type ppos52= patch_positions[52];
			idx_type ppos53= patch_positions[53];idx_type ppos54= patch_positions[54];idx_type ppos55= patch_positions[55];idx_type ppos56= patch_positions[56];
			idx_type ppos57= patch_positions[57];idx_type ppos58= patch_positions[58];idx_type ppos59= patch_positions[59];idx_type ppos60= patch_positions[60];
			idx_type ppos61= patch_positions[61];idx_type ppos62= patch_positions[62];idx_type ppos63= patch_positions[63];idx_type ppos64= patch_positions[64];
			idx_type ppos65= patch_positions[65];idx_type ppos66= patch_positions[66];idx_type ppos67= patch_positions[67];idx_type ppos68= patch_positions[68];
			idx_type ppos69= patch_positions[69];idx_type ppos70= patch_positions[70];idx_type ppos71= patch_positions[71];idx_type ppos72= patch_positions[72];
			idx_type ppos73= patch_positions[73];idx_type ppos74= patch_positions[74];idx_type ppos75= patch_positions[75];idx_type ppos76= patch_positions[76];
			idx_type ppos77= patch_positions[77];idx_type ppos78= patch_positions[78];idx_type ppos79= patch_positions[79];idx_type ppos80= patch_positions[80];
			idx_type ppos81= patch_positions[81];idx_type ppos82= patch_positions[82];idx_type ppos83= patch_positions[83];idx_type ppos84= patch_positions[84];
			idx_type ppos85= patch_positions[85];idx_type ppos86= patch_positions[86];idx_type ppos87= patch_positions[87];idx_type ppos88= patch_positions[88];
			idx_type ppos89= patch_positions[89];idx_type ppos90= patch_positions[90];idx_type ppos91= patch_positions[91];idx_type ppos92= patch_positions[92];
			idx_type ppos93= patch_positions[93];idx_type ppos94= patch_positions[94];idx_type ppos95= patch_positions[95];idx_type ppos96= patch_positions[96];
			idx_type ppos97= patch_positions[97];idx_type ppos98= patch_positions[98];idx_type ppos99= patch_positions[99];idx_type ppos100= patch_positions[100];
			idx_type ppos101= patch_positions[101];idx_type ppos102= patch_positions[102];idx_type ppos103= patch_positions[103];idx_type ppos104= patch_positions[104];
			idx_type ppos105= patch_positions[105];idx_type ppos106= patch_positions[106];idx_type ppos107= patch_positions[107];idx_type ppos108= patch_positions[108];
			idx_type ppos109= patch_positions[109];idx_type ppos110= patch_positions[110];idx_type ppos111= patch_positions[111];idx_type ppos112= patch_positions[112];
			idx_type ppos113= patch_positions[113];idx_type ppos114= patch_positions[114];

			//get patchvalues at origin (this is the fastest approach)
			float val_orig0 = image_previous[idx0        ];float val_orig1=image_previous[idx0+ppos1];float val_orig2=image_previous[idx0+ppos2];float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
			float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];float val_orig7=image_previous[idx0+ppos7];float val_orig8=image_previous[idx0+ppos8];
			float val_orig9=image_previous[idx0+ppos9];float val_orig10=image_previous[idx0+ppos10];float val_orig11=image_previous[idx0+ppos11];float val_orig12=image_previous[idx0+ppos12];
			float val_orig13=image_previous[idx0+ppos13];float val_orig14=image_previous[idx0+ppos14];float val_orig15=image_previous[idx0+ppos15];float val_orig16=image_previous[idx0+ppos16];
			float val_orig17=image_previous[idx0+ppos17];float val_orig18=image_previous[idx0+ppos18];float val_orig19=image_previous[idx0+ppos19];float val_orig20=image_previous[idx0+ppos20];
			float val_orig21=image_previous[idx0+ppos21];float val_orig22=image_previous[idx0+ppos22];float val_orig23=image_previous[idx0+ppos23];float val_orig24=image_previous[idx0+ppos24];
			float val_orig25=image_previous[idx0+ppos25];float val_orig26=image_previous[idx0+ppos26];float val_orig27=image_previous[idx0+ppos27];float val_orig28=image_previous[idx0+ppos28];
			float val_orig29=image_previous[idx0+ppos29];float val_orig30=image_previous[idx0+ppos30];float val_orig31=image_previous[idx0+ppos31];float val_orig32=image_previous[idx0+ppos32];
			float val_orig33=image_previous[idx0+ppos33];float val_orig34=image_previous[idx0+ppos34];float val_orig35=image_previous[idx0+ppos35];float val_orig36=image_previous[idx0+ppos36];
			float val_orig37=image_previous[idx0+ppos37];float val_orig38=image_previous[idx0+ppos38];float val_orig39=image_previous[idx0+ppos39];float val_orig40=image_previous[idx0+ppos40];
			float val_orig41=image_previous[idx0+ppos41];float val_orig42=image_previous[idx0+ppos42];float val_orig43=image_previous[idx0+ppos43];float val_orig44=image_previous[idx0+ppos44];
			float val_orig45=image_previous[idx0+ppos45];float val_orig46=image_previous[idx0+ppos46];float val_orig47=image_previous[idx0+ppos47];float val_orig48=image_previous[idx0+ppos48];
			float val_orig49=image_previous[idx0+ppos49];float val_orig50=image_previous[idx0+ppos50];float val_orig51=image_previous[idx0+ppos51];float val_orig52=image_previous[idx0+ppos52];
			float val_orig53=image_previous[idx0+ppos53];float val_orig54=image_previous[idx0+ppos54];float val_orig55=image_previous[idx0+ppos55];float val_orig56=image_previous[idx0+ppos56];
			float val_orig57=image_previous[idx0+ppos57];float val_orig58=image_previous[idx0+ppos58];float val_orig59=image_previous[idx0+ppos59];float val_orig60=image_previous[idx0+ppos60];
			float val_orig61=image_previous[idx0+ppos61];float val_orig62=image_previous[idx0+ppos62];float val_orig63=image_previous[idx0+ppos63];float val_orig64=image_previous[idx0+ppos64];
			float val_orig65=image_previous[idx0+ppos65];float val_orig66=image_previous[idx0+ppos66];float val_orig67=image_previous[idx0+ppos67];float val_orig68=image_previous[idx0+ppos68];
			float val_orig69=image_previous[idx0+ppos69];float val_orig70=image_previous[idx0+ppos70];float val_orig71=image_previous[idx0+ppos71];float val_orig72=image_previous[idx0+ppos72];
			float val_orig73=image_previous[idx0+ppos73];float val_orig74=image_previous[idx0+ppos74];float val_orig75=image_previous[idx0+ppos75];float val_orig76=image_previous[idx0+ppos76];
			float val_orig77=image_previous[idx0+ppos77];float val_orig78=image_previous[idx0+ppos78];float val_orig79=image_previous[idx0+ppos79];float val_orig80=image_previous[idx0+ppos80];
			float val_orig81=image_previous[idx0+ppos81];float val_orig82=image_previous[idx0+ppos82];float val_orig83=image_previous[idx0+ppos83];float val_orig84=image_previous[idx0+ppos84];
			float val_orig85=image_previous[idx0+ppos85];float val_orig86=image_previous[idx0+ppos86];float val_orig87=image_previous[idx0+ppos87];float val_orig88=image_previous[idx0+ppos88];
			float val_orig89=image_previous[idx0+ppos89];float val_orig90=image_previous[idx0+ppos90];float val_orig91=image_previous[idx0+ppos91];float val_orig92=image_previous[idx0+ppos92];
			float val_orig93=image_previous[idx0+ppos93];float val_orig94=image_previous[idx0+ppos94];float val_orig95=image_previous[idx0+ppos95];float val_orig96=image_previous[idx0+ppos96];
			float val_orig97=image_previous[idx0+ppos97];float val_orig98=image_previous[idx0+ppos98];float val_orig99=image_previous[idx0+ppos99];float val_orig100=image_previous[idx0+ppos100];
			float val_orig101=image_previous[idx0+ppos101];float val_orig102=image_previous[idx0+ppos102];float val_orig103=image_previous[idx0+ppos103];float val_orig104=image_previous[idx0+ppos104];
			float val_orig105=image_previous[idx0+ppos105];float val_orig106=image_previous[idx0+ppos106];float val_orig107=image_previous[idx0+ppos107];float val_orig108=image_previous[idx0+ppos108];
			float val_orig109=image_previous[idx0+ppos109];float val_orig110=image_previous[idx0+ppos110];float val_orig111=image_previous[idx0+ppos111];float val_orig112=image_previous[idx0+ppos112];
			float val_orig113=image_previous[idx0+ppos113];float val_orig114=image_previous[idx0+ppos114];

			float sigma = sigma_list[z0-zpad];
			float multiplier = -1.f/(sigma*sigma*beta);
			/////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////
			for (int s = 0; s < nsize_search; s++)
			{
				__syncthreads();

				idx_type idx1 = idx0 + search_positions[s];
				float noisy_value_searchpos = image_raw[idx1];

				//get patchvalues at search position
				/////////////////////////////////////////////////////////////////////////
				float distance = 0.0f;

				float tmp;
				tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos1]-val_orig1; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos2]-val_orig2; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos3]-val_orig3; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos4]-val_orig4; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos5]-val_orig5; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos6]-val_orig6; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos7]-val_orig7; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos8]-val_orig8; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos9]-val_orig9; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos10]-val_orig10; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos11]-val_orig11; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos12]-val_orig12; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos13]-val_orig13; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos14]-val_orig14; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos15]-val_orig15; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos16]-val_orig16; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos17]-val_orig17; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos18]-val_orig18; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos19]-val_orig19; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos20]-val_orig20; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos21]-val_orig21; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos22]-val_orig22; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos23]-val_orig23; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos24]-val_orig24; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos25]-val_orig25; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos26]-val_orig26; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos27]-val_orig27; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos28]-val_orig28; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos29]-val_orig29; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos30]-val_orig30; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos31]-val_orig31; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos32]-val_orig32; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos33]-val_orig33; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos34]-val_orig34; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos35]-val_orig35; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos36]-val_orig36; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos37]-val_orig37; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos38]-val_orig38; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos39]-val_orig39; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos40]-val_orig40; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos41]-val_orig41; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos42]-val_orig42; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos43]-val_orig43; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos44]-val_orig44; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos45]-val_orig45; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos46]-val_orig46; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos47]-val_orig47; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos48]-val_orig48; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos49]-val_orig49; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos50]-val_orig50; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos51]-val_orig51; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos52]-val_orig52; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos53]-val_orig53; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos54]-val_orig54; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos55]-val_orig55; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos56]-val_orig56; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos57]-val_orig57; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos58]-val_orig58; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos59]-val_orig59; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos60]-val_orig60; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos61]-val_orig61; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos62]-val_orig62; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos63]-val_orig63; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos64]-val_orig64; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos65]-val_orig65; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos66]-val_orig66; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos67]-val_orig67; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos68]-val_orig68; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos69]-val_orig69; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos70]-val_orig70; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos71]-val_orig71; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos72]-val_orig72; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos73]-val_orig73; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos74]-val_orig74; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos75]-val_orig75; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos76]-val_orig76; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos77]-val_orig77; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos78]-val_orig78; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos79]-val_orig79; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos80]-val_orig80; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos81]-val_orig81; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos82]-val_orig82; distance += (tmp*tmp)*0.111111f;
				tmp = image_previous[idx1 + ppos83]-val_orig83; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos84]-val_orig84; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos85]-val_orig85; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos86]-val_orig86; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos87]-val_orig87; distance += (tmp*tmp)*0.0682275f;
				tmp = image_previous[idx1 + ppos88]-val_orig88; distance += (tmp*tmp)*0.0501801f;
				tmp = image_previous[idx1 + ppos89]-val_orig89; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos90]-val_orig90; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos91]-val_orig91; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos92]-val_orig92; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos93]-val_orig93; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos94]-val_orig94; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos95]-val_orig95; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos96]-val_orig96; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos97]-val_orig97; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos98]-val_orig98; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos99]-val_orig99; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos100]-val_orig100; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos101]-val_orig101; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos102]-val_orig102; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos103]-val_orig103; distance += (tmp*tmp)*0.04f;
				tmp = image_previous[idx1 + ppos104]-val_orig104; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos105]-val_orig105; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos106]-val_orig106; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos107]-val_orig107; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos108]-val_orig108; distance += (tmp*tmp)*0.0333954f;
				tmp = image_previous[idx1 + ppos109]-val_orig109; distance += (tmp*tmp)*0.0287373f;
				tmp = image_previous[idx1 + ppos110]-val_orig110; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos111]-val_orig111; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos112]-val_orig112; distance += (tmp*tmp)*0.0225664f;
				tmp = image_previous[idx1 + ppos113]-val_orig113; distance += (tmp*tmp)*0.0204082f;
				tmp = image_previous[idx1 + ppos114]-val_orig114; distance += (tmp*tmp)*0.0204082f;
				/////////////////////////////////////////////////////////////////////////

				//weight the patch
				/////////////////////////////////////////////////////////////////////////
				distance = distance*multiplier;
				float this_weight = expf(distance); //this is faster on GPU than an own approximation

				filtervalue += this_weight*noisy_value_searchpos;
				filterweight += this_weight;

				maxweight = ((this_weight > maxweight) ? this_weight : maxweight);
				/////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////////////////////////////////

			if (maxweight > 0.0f)
			{
				filtervalue += maxweight*noisy_value_origin;
				filterweight += maxweight;

				filtervalue /= filterweight;
			}
			else
				filtervalue = noisy_value_origin;

			__syncthreads();
			next_result[idx_unpadded] = filtervalue;

			return;
		}
	}

	int IterativeNLM_GPU::configure_device(int shape[3], protocol::DenoiseParameters *params)
	{
		int devicecount = 0;
		cudaGetDeviceCount(&devicecount);
		if(devicecount < 1) {params->gpu.n_gpus = 0; return -1;}
		else params->gpu.n_gpus = std::min(devicecount-params->gpu.deviceID, params->gpu.n_gpus);

		deviceID = params->gpu.deviceID;
		ngpus = params->gpu.n_gpus;
		threadsPerBlock = params->gpu.threadsPerBlock;
		cudaSetDevice(deviceID);

		//Unrolling does not improve speed on GPU
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		padding[0] = params->radius_searchspace[0]+params->radius_patchspace[0];
		padding[1] = params->radius_searchspace[1]+params->radius_patchspace[1];
		padding[2] = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];
		padding[3] = params->radius_searchspace[0]+params->radius_patchspace[0];
		padding[4] = params->radius_searchspace[1]+params->radius_patchspace[1];
		padding[5] = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		shape_padded[0] = shape[0]+padding[0]+padding[3];
		shape_padded[1] = shape[1]+padding[1]+padding[4];
		shape_padded[2] = shape[2]+padding[2]+padding[5];
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//precalculate immutable values
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		idx_type *search_positions_host = setup_searchspace(shape_padded, params, nsize_search);
		idx_type *patch_positions_host = setup_patchspace(shape_padded, params, nsize_patch);
		float *distweight_host = setup_distweight(shape_padded, params);

		//pointer arrays for multi GPU
		search_positions = new idx_type*[params->gpu.n_gpus];
		patch_positions  = new idx_type*[params->gpu.n_gpus];
		distweight       = new float*[params->gpu.n_gpus];
		sigma_list       = new float*[params->gpu.n_gpus];

		for (int gpu = 0; gpu < params->gpu.n_gpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);

			(idx_type*) cudaMalloc((void**)&search_positions[gpu], nsize_search*sizeof(*search_positions[gpu]));
			(idx_type*) cudaMalloc((void**)&patch_positions[gpu], nsize_patch*sizeof(*patch_positions[gpu]));
			(float*) cudaMalloc((void**)&distweight[gpu], nsize_patch*sizeof(*distweight[gpu]));
			(float*) cudaMalloc((void**)&sigma_list[gpu], shape[2]*sizeof(*sigma_list[gpu]));

			cudaMemcpy(search_positions[gpu], search_positions_host, nsize_search*sizeof(*search_positions[gpu]), cudaMemcpyHostToDevice);
			cudaMemcpy(patch_positions[gpu], patch_positions_host, nsize_patch*sizeof(*patch_positions[gpu]), cudaMemcpyHostToDevice);
			cudaMemcpy(distweight[gpu], distweight_host, nsize_patch*sizeof(*distweight[gpu]), cudaMemcpyHostToDevice);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		//generate kernel code
		if (0 == 1){
			std::ofstream outfile;
		    outfile.open("/home/stefan/Documents/kernelcode.txt");

		    outfile << "patchsize: " << nsize_patch << "\n";

			for (int i = 0; i < nsize_patch; i++)
			{
				if (i == 0) continue;
				else outfile << "idx_type ppos"<<i<< "= patch_positions["<<i<<"];";
				if (i != 0 && (i%4) == 0) outfile << "\n";
			}
			outfile << "\n-----------------------------\n";
			for (int i = 0; i < nsize_patch; i++)
			{
				if (i == 0)outfile << "float val_orig0 = image_previous[idx0        ];";
				else outfile << "float val_orig"<<i<<"=image_previous[idx0+ppos"<<i<<"];";
				if (i != 0 && (i%4) == 0) outfile << "\n";
			}
			outfile << "\n-----------------------------\n";
			for (int i = 0; i < nsize_patch; i++)
			{
				if (i== 0)outfile << "tmp = image_previous[idx1        ]-val_orig0; distance += (tmp*tmp)*" << distweight_host[i] << "f;\n";
				else      outfile<< "tmp = image_previous[idx1 + ppos" << i << "]-val_orig"<<i<<"; distance += (tmp*tmp)*" << distweight_host[i] << "f;\n";
			}
			outfile << "\n-----------------------------\n";
			outfile.close();
		}

		free(search_positions_host);
		free(patch_positions_host);
		free(distweight_host);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//estimate how many slices can be denoised at once
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		n_parallelslices = shape[2]/ngpus;
		if((shape[2]%ngpus) != 0) n_parallelslices++;

		long long int nslice = shape[0]*shape[1];
		long long int nstack = n_parallelslices*nslice;

		long long int nslice_padded = shape_padded[0]*shape_padded[1];
		long long int nstack_padded = (n_parallelslices+padding[2]+padding[5])*nslice_padded;

		size_t free_byte, total_byte ;
		double free_db;

		for (int gpu = 0; gpu < params->gpu.n_gpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);
			cudaMemGetInfo( &free_byte, &total_byte ) ;

			if (gpu == 0)
				free_db = (double)free_byte-params->gpu.memory_buffer*free_db; //subtract some memory to be kept free
			else
				free_db = std::min(free_db, (double)free_byte-params->gpu.memory_buffer*free_db);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		double expected_usage = (2*nstack_padded+nstack)*sizeof(float);

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, deviceID);

		while (expected_usage > free_db || nstack_padded*sizeof(float) > props.maxGridSize[0])
		{
			n_parallelslices--;
			nstack_padded -= nslice_padded;
			nstack -= nslice;

			expected_usage = (2*nstack_padded+nstack)*sizeof(float);
		}
		////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//allocate sufficient gpu-memory
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		idx_type asize1 = nstack_padded*sizeof(*image_raw[0]);
		idx_type asize2 = nstack_padded*sizeof(*image_previous[0]);
		idx_type asize3 = nstack       *sizeof(*next_result[0]);

		image_raw      = new float*[params->gpu.n_gpus];
		image_previous = new float*[params->gpu.n_gpus];
		next_result    = new float*[params->gpu.n_gpus];

		for (int gpu = 0; gpu < params->gpu.n_gpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);
			(float*) cudaMalloc((void**)&image_raw[gpu], asize1*sizeof(*image_raw[gpu]));
			(float*) cudaMalloc((void**)&image_previous[gpu], asize2*sizeof(*image_previous[gpu]));
			(float*) cudaMalloc((void**)&next_result[gpu], asize3*sizeof(*next_result[gpu]));
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//copy immutable denoise parameters to constant memory
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int gpu = 0; gpu < params->gpu.n_gpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);
			cudaMemcpyToSymbol(gpu_const::nslices_searchspace, &params->nslices, sizeof(gpu_const::nslices_searchspace));
			cudaMemcpyToSymbol(gpu_const::beta, &params->beta, sizeof(gpu_const::beta));
			cudaMemcpyToSymbol(gpu_const::nsize_patch, &nsize_patch, sizeof(gpu_const::nsize_patch));
			cudaMemcpyToSymbol(gpu_const::nsize_search, &nsize_search, sizeof(gpu_const::nsize_search));

			cudaMemcpyToSymbol(gpu_const::radius_searchspace, &params->radius_searchspace,  3*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::radius_patchspace, &params->radius_patchspace,  3*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::padding, &padding,  6*sizeof(int), 0);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);

			std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
			if (error_string != "no error")
			{
				std::cout << "Device Configuration: " << error_string << std::endl;
				return -2;
			}
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		cudaSetDevice(deviceID);
		return n_parallelslices;
	}
	void IterativeNLM_GPU::free_device()
	{
		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);

			cudaFree(search_positions[gpu]);
			cudaFree(patch_positions[gpu]);
			cudaFree(distweight[gpu]);
			cudaFree(sigma_list[gpu]);
			cudaFree(image_raw[gpu]);
			cudaFree(image_previous[gpu]);
			cudaFree(next_result[gpu]);
		}
		cudaSetDevice(deviceID);

		free(search_positions);
		free(patch_positions);
		free(distweight);
		free(sigma_list);
		free(image_raw);
		free(image_previous);
		free(next_result);

		return;
	}

	void IterativeNLM_GPU::Run_GaussianNoise(int iter, float* instack, int shape[3], protocol::DenoiseParameters *params)
	{
		// The straightforward method for small stacks:
		//     puts all slices on a single GPU and denoises in one pass

		cudaSetDevice(deviceID);
		auto time0 = std::chrono::high_resolution_clock::now();

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);

		//Check if already fully processed and just push result to device
		/////////////////////////////////////////////////////////////////////////////
		if (params->io.resume)
		{
			bool finished_block = true;
			struct stat buffer;

			for (int i = params->io.firstslice; i <= params->io.lastslice; i++)
			{
				//file exists?
				std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i,4)+".tif";
				if (stat (filename.c_str(), &buffer) != 0){finished_block = false; break;}

				//file is complete?
				std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
				const auto filesize = testFile.tellg();
				if (filesize > expected_filesize) expected_filesize = filesize;
				else if (filesize != expected_filesize){finished_block = false; break;}
			}
			if (finished_block)
			{
				hdcom::HdCommunication hdcom;
				std::vector<std::string> filelist = hdcom.GetFilelist(params->io.active_outpath, shape);
				float *result = hdcom.Get3DTifSequence_32bitPointer(filelist,shape,params->io.firstslice,params->io.lastslice);
				cudaMemcpy(next_result[0], result, nstack*sizeof(*next_result[0]), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				free(result);

				if (iter == 1) prepare_iteration1(instack, shape);

				std::cout << "iteration " << iter << " read in from disk" << std::endl;
				return;
			}
			resumed = finished_block;
		}
		/////////////////////////////////////////////////////////////////////////////

		if (iter == 1) prepare_iteration1(instack, shape);
		else prepare_nextiteration(shape);

		if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
			gpu_denoise::apply_filter_patch111<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
			gpu_denoise::apply_filter_patch221<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
			gpu_denoise::apply_filter_patch222<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 1)
			gpu_denoise::apply_filter_patch331<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 2)
			gpu_denoise::apply_filter_patch332<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 3)
			gpu_denoise::apply_filter_patch333<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
			gpu_denoise::apply_filter_patch112<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
			gpu_denoise::apply_filter_patch113<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0]);
		else
			gpu_denoise::apply_filter_generic<<<blocksPerGrid,threadsPerBlock>>>(image_raw[0], image_previous[0], next_result[0], sigma_list[0], search_positions[0], patch_positions[0], distweight[0]);
		cudaDeviceSynchronize();

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << elapsed_total.count() << " s                          " << std::endl;

		return;
	}
	void IterativeNLM_GPU::Run_GaussianNoise_GPUBlocks(int iter, float* instack, float * &previous, int shape[3], protocol::DenoiseParameters *params)
	{
		// The method for machines with plenty of RAM:
		//     puts all slices in RAM and splits denoising on the GPU by compute capacity
		//     (allows for multiGPU denoising)

		cudaSetDevice(deviceID);
		auto time0 = std::chrono::high_resolution_clock::now();

		int blocklength = n_parallelslices;
		int n_blocks = shape[2]/blocklength;
		if ((shape[2]%blocklength) != 0) n_blocks++;

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		//Check if already fully processed and just push result to device
		/////////////////////////////////////////////////////////////////////////////
		if (params->io.resume)
		{
			bool finished_block = true;
			struct stat buffer;

			for (int i = params->io.firstslice; i <= params->io.lastslice; i++)
			{
				//file exists?
				std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i,4)+".tif";
				if (stat (filename.c_str(), &buffer) != 0){finished_block = false; break;}

				//file is complete?
				std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
				const auto filesize = testFile.tellg();
				if (filesize > expected_filesize) expected_filesize = filesize;
				else if (filesize != expected_filesize){finished_block = false; break;}
			}
			if (finished_block)
			{
				hdcom::HdCommunication hdcom;
				std::vector<std::string> filelist = hdcom.GetFilelist(params->io.active_outpath, shape);
				free(previous);
				previous = hdcom.Get3DTifSequence_32bitPointer(filelist,shape,params->io.firstslice,params->io.lastslice);

				std::cout << "iteration " << iter << " read in from disk" << std::endl;
				return;
			}
			resumed = finished_block;
		}
		/////////////////////////////////////////////////////////////////////////////

		int* firstslice = new int[ngpus];
		int* lastslice = new int[ngpus];

		for (int n = 0; n < n_blocks; n+=ngpus)
		{
			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				firstslice[gpu] = std::min((n+gpu)*blocklength, shape[2]-1);
				lastslice[gpu] = std::min((n+1+gpu)*blocklength-1, shape[2]-1);
			}

			if (iter == 1) prepare_iteration1_block(instack, shape, firstslice, lastslice);
			else prepare_nextiteration_block(instack, previous, shape, firstslice, lastslice);

			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				nstack = ((lastslice[gpu]+1)-firstslice[gpu])*nslice;
				int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);

				if (blocksPerGrid > 0)
				{
					cudaSetDevice(deviceID+gpu);

				if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch111<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch221<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch222<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch331<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch332<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 3)
					gpu_denoise::apply_filter_patch333<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch112<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
					gpu_denoise::apply_filter_patch113<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else
					gpu_denoise::apply_filter_generic<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu], distweight[gpu]);

				}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				cudaSetDevice(deviceID+gpu);

				std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
				if (error_string != "no error")
				{
					std::cout << "Block " << n << ", GPU " << gpu << " section2: " << error_string << std::endl;
					return;
				}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			//get result
			///////////////////////////////////////////////////////////////////////////////////////////
			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				int blockshape[3] = {shape[0], shape[1], ((lastslice[gpu]+1)-firstslice[gpu])};

				cudaSetDevice(deviceID+gpu);

				idx_type nstack_block = nslice*blockshape[2];
				idx_type asize1 = nstack_block*sizeof(*next_result[0]);

				if(asize1 > 0) cudaMemcpyAsync(previous+(firstslice[gpu]*nslice),next_result[gpu], asize1, cudaMemcpyDeviceToHost);
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}
			///////////////////////////////////////////////////////////////////////////////////////////

			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				cudaSetDevice(deviceID+gpu);

				std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
				if (error_string != "no error")
				{
					std::cout << "Block " << n << ", GPU " << gpu << ": " << error_string << std::endl;
					return;
				}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			auto time_final = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_total = time_final-time0;
			std::cout << "iteration " << iter << ": " << std::min(shape[2], (n+ngpus)*n_parallelslices) << "/" << shape[2] << ", "
					<< round(elapsed_total.count()/(n+ngpus)*10.)/(10.f*n_parallelslices*ngpus)*
					  (shape[2]-std::min(shape[2], (n+ngpus)*n_parallelslices))*ngpus
					<< " s remaining          \r";
			std::cout.flush();
			////////////////////////////////////////////////////////////////////////////////////////////
		}

		free(firstslice);
		free(lastslice);

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << elapsed_total.count() << " s                          " << std::endl;

		return;
	}
	void IterativeNLM_GPU::Run_GaussianNoise_GPUBlocks(int iter, std::vector<std::string> &filelist_raw, std::vector<std::string> &filelist_prev, int shape[3], protocol::DenoiseParameters *params)
	{
		//The method for insufficient memory:
		//    reads in what is necessary to fully occupy the GPU
		//

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);
			cudaMemcpyToSymbol(gpu_const::nx, &shape[0], sizeof(gpu_const::nx));
			cudaMemcpyToSymbol(gpu_const::ny, &shape[1], sizeof(gpu_const::ny));
		}

		//check if only substack processing and get offset in filelists
		/////////////////////////////////////////////////////////////////
		int firstslice_offset[2] = {0, 0};

		if (params->io.firstslice != 0)
		{
			firstslice_offset[0] = params->io.firstslice;

			std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";

			for (int i = 0; i < filelist_prev.size(); i++)
				if (filelist_prev[i].substr(filelist_prev[i].length()-8,8) == fs_sub){firstslice_offset[1] = i; break;}
		}
		/////////////////////////////////////////////////////////////////

		//set the block dimensions
		/////////////////////////////////////////////////////////////////
		int blocklength = n_parallelslices;
		int n_blocks = shape[2]/blocklength;
		if ((shape[2]%blocklength) != 0) n_blocks++;

		padding[0] = params->radius_searchspace[0]+params->radius_patchspace[0];
		padding[1] = params->radius_searchspace[1]+params->radius_patchspace[1];
		padding[2] = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];
		padding[3] = params->radius_searchspace[0]+params->radius_patchspace[0];
		padding[4] = params->radius_searchspace[1]+params->radius_patchspace[1];
		padding[5] = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int zpadding[2] = {padding[2], padding[5]};

		int blockshape[3] = {shape[0], shape[1], blocklength*ngpus};
		long long int blockslice = blockshape[0]*blockshape[1];
		long long int blockstack = blockshape[2]*blockslice;

		int blockshape_padded[3] = {blockshape[0]+padding[0]+padding[3], blockshape[1]+padding[1]+padding[4], blockshape[2]+padding[2]+padding[5]};
		idx_type blockslice_padded = blockshape_padded[0]*blockshape_padded[1];
		/////////////////////////////////////////////////////////////////

		float* output = (float*) malloc(blockstack*sizeof(*output));

		hdcom::HdCommunication hdcom;
		auto time0 = std::chrono::high_resolution_clock::now();
		int skipped_blocks = 0;

		for (int n = 0; n < n_blocks; n+=ngpus)
		{
			int firstslice = n*blocklength-zpadding[0];
			int lastslice = std::min((n+ngpus)*blocklength-1, shape[2]-1)+zpadding[1];

			//Check if block already fully processed and resume
			/////////////////////////////////////////////////////////////////////////////
			if (params->io.resume)
			{
				bool finished_block = true;
				struct stat buffer;

				for (int i = firstslice+zpadding[0]; i <= lastslice-zpadding[1]; i++)
				{
					//file exists?
					std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i+params->io.firstslice,4)+".tif";
					if (stat (filename.c_str(), &buffer) != 0){finished_block = false; break;}

					//file is complete?
					std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
					const auto filesize = testFile.tellg();
					if (filesize > expected_filesize) expected_filesize = filesize;
					else if (filesize != expected_filesize){finished_block = false; break;}
				}
				if (finished_block){skipped_blocks++; continue;}
			}
			/////////////////////////////////////////////////////////////////////////////

			//check if we need to pad in z-direction
			/////////////////////////////////////////////////////////////////////////////
			if (firstslice < 0) {padding[2] = -firstslice; firstslice = 0;}
			else padding[2] = 0;
			if (lastslice >= shape[2]) {padding[5] = lastslice-(shape[2]-1); lastslice = shape[2]-1;}
			else padding[5] = 0;
			/////////////////////////////////////////////////////////////////////////////

			//apply padding on host
			/////////////////////////////////////////////////////////////////////////////
			float *imageblock_raw, *imageblock_previous;

			if (iter == 1 && lastslice > firstslice)
				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
			else if (lastslice > firstslice)
			{
				float* tmp = hdcom.Get3DTifSequence_32bitPointer(filelist_prev, blockshape, firstslice+firstslice_offset[1], lastslice+firstslice_offset[1]);
				imageblock_previous = pad_reflective(tmp, padding, blockshape, blockshape_padded);

				free(tmp);

				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
			}
			if (lastslice > firstslice)
			{
				float* tmp = pad_reflective(imageblock_raw, padding, blockshape, blockshape_padded);

				std::swap(tmp, imageblock_raw);
				free(tmp);
			}
			/////////////////////////////////////////////////////////////////////////////

			//prepare device
			/////////////////////////////////////////////////////////////////////////////
			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				int blockdim2 = std::min((n+gpu+1)*blocklength, shape[2])-((n+gpu)*blocklength);
				long long int asize1 = ((blockdim2+zpadding[0]+zpadding[1])*blockslice_padded)*sizeof(*image_raw[gpu]);

				cudaSetDevice(deviceID+gpu);
				cudaMemcpyToSymbol(gpu_const::nz, &blockdim2, sizeof(gpu_const::nz));

				if(iter == 1){
					if(asize1 > 0)
					cudaMemcpyAsync(image_raw[gpu], imageblock_raw+(gpu*blocklength)*blockslice_padded, asize1, cudaMemcpyHostToDevice);}
				else{
					if(asize1 > 0){
					cudaMemcpyAsync(image_raw[gpu], imageblock_raw+(gpu*blocklength)*blockslice_padded, asize1, cudaMemcpyHostToDevice);
					cudaMemcpyAsync(image_previous[gpu], imageblock_previous+(gpu*blocklength)*blockslice_padded, asize1, cudaMemcpyHostToDevice);}}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			if (iter == 1){
				for (int gpu = 0; gpu < ngpus; gpu++)
				{
					int blockdim2 = std::min((n+gpu+1)*blocklength, shape[2])-((n+gpu)*blocklength);
					long long int asize1 = ((blockdim2+zpadding[0]+zpadding[1])*blockslice_padded)*sizeof(*image_raw[gpu]);

					cudaSetDevice(deviceID+gpu);
					if(asize1 > 0) cudaMemcpyAsync(image_previous[gpu], image_raw[gpu], asize1, cudaMemcpyDeviceToDevice);
				}
				for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}
			}

			if (lastslice > firstslice && iter > 1)free(imageblock_previous);
			if (lastslice > firstslice)free(imageblock_raw);
			/////////////////////////////////////////////////////////////////////////////

			//filter on device
			/////////////////////////////////////////////////////////////////////////////
			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				int blockdim2 = std::min((n+gpu+1)*blocklength, shape[2])-((n+gpu)*blocklength);
				idx_type blockstack = (blockdim2*blockslice);
				int blocksPerGrid = (blockstack + threadsPerBlock - 1) / (threadsPerBlock);

				if (blocksPerGrid > 0)
				{
					cudaSetDevice(deviceID+gpu);

				if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch111<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch221<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch222<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 1)
					gpu_denoise::apply_filter_patch331<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch332<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 3)
					gpu_denoise::apply_filter_patch333<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
					gpu_denoise::apply_filter_patch112<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
					gpu_denoise::apply_filter_patch113<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu]);
				else
					gpu_denoise::apply_filter_generic<<<blocksPerGrid,threadsPerBlock>>>(image_raw[gpu], image_previous[gpu], next_result[gpu], sigma_list[gpu], search_positions[gpu], patch_positions[gpu], distweight[gpu]);
				}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			for (int gpu = 0; gpu < ngpus; gpu++)
			{
				int blockdim2 = std::min((n+gpu+1)*blocklength, shape[2])-((n+gpu)*blocklength);

				if(gpu*blocklength < shape[2] && blockdim2 > 0)
				{
					cudaSetDevice(deviceID+gpu);

					idx_type this_nstack = blockslice*blockdim2;
					idx_type asize1 = this_nstack*sizeof(*next_result[gpu]);

					cudaMemcpyAsync(output+(gpu*blocklength)*blockslice,next_result[gpu], asize1, cudaMemcpyDeviceToHost);
				}
			}
			for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

			/////////////////////////////////////////////////////////////////////////////
			blockshape[2] = std::min(blocklength*ngpus, params->io.lastslice+1-((n*blocklength)+params->io.firstslice));

			if(params->io.save_type == "16bit") hdcom.SaveTifSequence_as16bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);
			else hdcom.SaveTifSequence_32bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			auto time_final = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_total = time_final-time0;
			std::cout << "iteration " << iter << ": " << std::min(shape[2], (n+ngpus)*n_parallelslices) << "/" << shape[2] << ", "
					<< round(elapsed_total.count()/(n+ngpus-skipped_blocks)*10.)/(10.f*n_parallelslices*ngpus)*
					  (shape[2]-std::min(shape[2]-skipped_blocks*blocklength, (n+ngpus)*n_parallelslices-skipped_blocks*blocklength))*ngpus
					<< " s remaining          \r";
			std::cout.flush();
			////////////////////////////////////////////////////////////////////////////////////////////
		}
		//////////////////////////////////////////////////////////////////////////////
		free(output);

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << elapsed_total.count() << " s                          " << std::endl;

		return;
	}

	void IterativeNLM_GPU::prepare_iteration1(float* input, int shape[3])
	{
		//
		//overwrites next_result and applies padding!
		//

		//////////////////////////////////////////////////////////////////////
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int asize1 = nstack*sizeof(*next_result[0]);

		cudaSetDevice(deviceID);
		cudaMemcpyToSymbol(gpu_const::nx, &shape[0], sizeof(gpu_const::nx));
		cudaMemcpyToSymbol(gpu_const::ny, &shape[1], sizeof(gpu_const::ny));
		cudaMemcpyToSymbol(gpu_const::nz, &shape[2], sizeof(gpu_const::nz));

		cudaMemcpy(next_result[0], input, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		//////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////
		long long int nslice_padded = (shape[0]+padding[0]+padding[3])*(shape[1]+padding[1]+padding[4]);
		long long int nstack_padded = (shape[2]+padding[2]+padding[5])*nslice_padded;

		int blocksPerGrid = (nstack_padded + threadsPerBlock - 1) / (threadsPerBlock);

		gpu_denoise::pad_reflective<<<blocksPerGrid,threadsPerBlock>>>(next_result[0], image_raw[0]);
		cudaDeviceSynchronize();
		//////////////////////////////////////////////////////////////////////

		asize1 = nstack_padded*sizeof(*image_previous[0]);
		cudaMemcpy(image_previous[0], image_raw[0], asize1, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void IterativeNLM_GPU::prepare_iteration1_block(float* input, int shape[3], const int* firstslice, const int* lastslice)
	{
		idx_type nslice = shape[0]*shape[1];
		long long int nslice_padded = (shape[0]+padding[0]+padding[3])*(shape[1]+padding[1]+padding[4]);

		int* block_dim2 = new int[ngpus];
		int* zpadding = new int[2*ngpus];
		int* initialdepth = new int[ngpus];

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			initialdepth[gpu] = (lastslice[gpu]+1)-firstslice[gpu];
			int this_padding[6] = {padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]};

			//check how we need to pad in z
			//////////////////////////////////////////////////////////////////////
			int active_firstslice = firstslice[gpu]-padding[2];
			int active_lastslice = lastslice[gpu] + padding[5];

			if (active_firstslice < 0) {this_padding[2] = -active_firstslice; active_firstslice = 0;}
			else this_padding[2] = 0;

			if (active_lastslice >= shape[2]) {this_padding[5] = active_lastslice-(shape[2]-1); active_lastslice = shape[2]-1;}
			else this_padding[5] = 0;

			zpadding[2*gpu] = this_padding[2];
			zpadding[2*gpu+1] = this_padding[5];

			block_dim2[gpu] = (active_lastslice+1)-active_firstslice;

			idx_type offset = active_firstslice*nslice;
			idx_type nstack = block_dim2[gpu]*nslice;
			long long int asize1 = nstack*sizeof(*next_result[gpu]);
			//////////////////////////////////////////////////////////////////////

			cudaSetDevice(deviceID+gpu);
			cudaMemcpyToSymbol(gpu_const::nz, &block_dim2[gpu], sizeof(gpu_const::nz));
			cudaMemcpyToSymbol(gpu_const::nx, &shape[0], sizeof(gpu_const::nx));
			cudaMemcpyToSymbol(gpu_const::ny, &shape[1], sizeof(gpu_const::ny));
			cudaMemcpyToSymbol(gpu_const::padding, &this_padding,  6*sizeof(int), 0); //update padding
			if (asize1 > 0)
				cudaMemcpyAsync(next_result[gpu], input+offset, asize1, cudaMemcpyHostToDevice);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			long long int nstack_padded = (block_dim2[gpu]+zpadding[2*gpu]+zpadding[2*gpu+1])*nslice_padded;
			int blocksPerGrid = (nstack_padded + threadsPerBlock - 1) / (threadsPerBlock);

			cudaSetDevice(deviceID+gpu);
			if (blocksPerGrid > 0)
				gpu_denoise::pad_reflective<<<blocksPerGrid,threadsPerBlock>>>(next_result[gpu], image_raw[gpu]);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			long long int nstack_padded = (block_dim2[gpu]+zpadding[2*gpu]+zpadding[2*gpu+1])*nslice_padded;
			long long int asize1 = nstack_padded*sizeof(*image_previous[gpu]);

			cudaSetDevice(deviceID+gpu);
			if(asize1 > 0) cudaMemcpyAsync(image_previous[gpu], image_raw[gpu], asize1, cudaMemcpyDeviceToDevice);

			//reset
			cudaMemcpyToSymbol(gpu_const::nz, &initialdepth[gpu], sizeof(gpu_const::nz));
			cudaMemcpyToSymbol(gpu_const::padding, &padding,  6*sizeof(int), 0);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		free(block_dim2);
		free(zpadding);
		free(initialdepth);

		cudaSetDevice(deviceID);
		return;
	}
	void IterativeNLM_GPU::prepare_nextiteration(int shape[3])
	{
		//////////////////////////////////////////////////////////////////////
		long long int nslice_padded = (shape[0]+padding[0]+padding[3])*(shape[1]+padding[1]+padding[4]);
		long long int nstack_padded = (shape[2]+padding[2]+padding[5])*nslice_padded;

		int blocksPerGrid = (nstack_padded + threadsPerBlock - 1) / (threadsPerBlock);

		cudaSetDevice(deviceID);
		gpu_denoise::pad_reflective<<<blocksPerGrid,threadsPerBlock>>>(next_result[0], image_previous[0]);
		cudaDeviceSynchronize();
		//////////////////////////////////////////////////////////////////////
		return;
	}
	void IterativeNLM_GPU::prepare_nextiteration_block(float* input, float *prev_result, int shape[3], const int* firstslice, const int* lastslice)
	{
		idx_type nslice = shape[0]*shape[1];
		long long int nslice_padded = (shape[0]+padding[0]+padding[3])*(shape[1]+padding[1]+padding[4]);

		int* block_dim2 = new int[ngpus];
		int* zpadding = new int[2*ngpus];
		int* initialdepth = new int[ngpus];
		long long int* offset = new long long int[ngpus];

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			initialdepth[gpu] = (lastslice[gpu]+1)-firstslice[gpu];
			int this_padding[6] = {padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]};

			//check how we need to pad in z
			//////////////////////////////////////////////////////////////////////
			int active_firstslice = firstslice[gpu]-padding[2];
			int active_lastslice = lastslice[gpu] + padding[5];

			if (active_firstslice < 0) {this_padding[2] = -active_firstslice; active_firstslice = 0;}
			else this_padding[2] = 0;

			if (active_lastslice >= shape[2]) {this_padding[5] = active_lastslice-(shape[2]-1); active_lastslice = shape[2]-1;}
			else this_padding[5] = 0;

			zpadding[2*gpu] = this_padding[2];
			zpadding[2*gpu+1] = this_padding[5];

			block_dim2[gpu] = (active_lastslice+1)-active_firstslice;

			offset[gpu] = active_firstslice*nslice;
			idx_type nstack = block_dim2[gpu]*nslice;
			long long int asize1 = nstack*sizeof(*next_result[gpu]);
			//////////////////////////////////////////////////////////////////////

			cudaSetDevice(deviceID+gpu);
			cudaMemcpyToSymbol(gpu_const::nz, &block_dim2[gpu], sizeof(gpu_const::nz));
			cudaMemcpyToSymbol(gpu_const::nx, &shape[0], sizeof(gpu_const::nx));
			cudaMemcpyToSymbol(gpu_const::ny, &shape[1], sizeof(gpu_const::ny));
			cudaMemcpyToSymbol(gpu_const::padding, &this_padding,  6*sizeof(int), 0); //update padding

			if (asize1 > 0)
				cudaMemcpyAsync(next_result[gpu], input+offset[gpu], asize1, cudaMemcpyHostToDevice);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			long long int nstack_padded = (block_dim2[gpu]+zpadding[2*gpu]+zpadding[2*gpu+1])*nslice_padded;
			int blocksPerGrid = (nstack_padded + threadsPerBlock - 1) / (threadsPerBlock);

			cudaSetDevice(deviceID+gpu);
			if (blocksPerGrid > 0)
				gpu_denoise::pad_reflective<<<blocksPerGrid,threadsPerBlock>>>(next_result[gpu], image_raw[gpu]);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			idx_type nstack = block_dim2[gpu]*nslice;
			long long int asize1 = nstack*sizeof(*next_result[gpu]);

			cudaSetDevice(deviceID+gpu);
			cudaMemcpyAsync(next_result[gpu], prev_result+offset[gpu], asize1, cudaMemcpyHostToDevice);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			long long int nstack_padded = (block_dim2[gpu]+zpadding[2*gpu]+zpadding[2*gpu+1])*nslice_padded;
			int blocksPerGrid = (nstack_padded + threadsPerBlock - 1) / (threadsPerBlock);

			cudaSetDevice(deviceID+gpu);
			if (blocksPerGrid > 0)
				gpu_denoise::pad_reflective<<<blocksPerGrid,threadsPerBlock>>>(next_result[gpu], image_previous[gpu]);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			//reset
			cudaSetDevice(deviceID+gpu);
			cudaMemcpyToSymbol(gpu_const::nz, &initialdepth[gpu], sizeof(gpu_const::nz));
			cudaMemcpyToSymbol(gpu_const::padding, &padding,  6*sizeof(int), 0);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		free(block_dim2);
		free(zpadding);
		free(initialdepth);
		free(offset);
		cudaSetDevice(deviceID);

		return;
	}

	void IterativeNLM_GPU::set_sigma(float* sigmalist, int shape[3])
	{
		int nz = shape[2];
		long long int asize1 = nz*sizeof(*sigma_list[0]);

		for (int gpu = 0; gpu < ngpus; gpu++)
		{
			cudaSetDevice(deviceID+gpu);
			cudaMemcpyAsync(sigma_list[gpu], sigmalist, asize1, cudaMemcpyHostToDevice);
		}
		for (int gpu = 0; gpu < ngpus; gpu++){cudaSetDevice(deviceID+gpu);cudaDeviceSynchronize();}

		cudaSetDevice(deviceID);
		return;
	}
	void IterativeNLM_GPU::get_result(float* output, int shape[3])
	{
		cudaSetDevice(deviceID);

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = nstack*sizeof(*next_result[0]);

		cudaMemcpy(output,next_result[0], asize1, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		return;
	}
}

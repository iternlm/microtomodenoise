#include "iternlm_cpu.h"

//Contains precalculated kernels for common patch sizes

namespace denoise
{
	void IterativeNLM_CPU::filterslice_p111(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		idx_type ppos1 =  patch_positions[1];
		idx_type ppos2 =  patch_positions[2];
		idx_type ppos3 =  patch_positions[3];
		idx_type ppos4 =  patch_positions[4];
		idx_type ppos5 =  patch_positions[5];
		idx_type ppos6 =  patch_positions[6];

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0];
				float val_orig1 = image_previous[idx0 + ppos1];
				float val_orig2 = image_previous[idx0 + ppos2];
				float val_orig3 = image_previous[idx0 + ppos3];
				float val_orig4 = image_previous[idx0 + ppos4];
				float val_orig5 = image_previous[idx0 + ppos5];
				float val_orig6 = image_previous[idx0 + ppos6];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p112(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		idx_type ppos1 =  patch_positions[1];
		idx_type ppos2 =  patch_positions[2];
		idx_type ppos3 =  patch_positions[3];
		idx_type ppos4 =  patch_positions[4];
		idx_type ppos5 =  patch_positions[5];
		idx_type ppos6 =  patch_positions[6];
		idx_type ppos7 =  patch_positions[7];
		idx_type ppos8 =  patch_positions[8];

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0        ];
				float val_orig1=image_previous[idx0+ppos1];float val_orig2=image_previous[idx0+ppos2];
				float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
				float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];float val_orig7=image_previous[idx0+ppos7];
				float val_orig8=image_previous[idx0+ppos8];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p113(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

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

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0        ];float val_orig1=image_previous[idx0+ppos1];
				float val_orig2=image_previous[idx0+ppos2];float val_orig3=image_previous[idx0+ppos3];float val_orig4=image_previous[idx0+ppos4];
				float val_orig5=image_previous[idx0+ppos5];float val_orig6=image_previous[idx0+ppos6];
				float val_orig7=image_previous[idx0+ppos7];float val_orig8=image_previous[idx0+ppos8];
				float val_orig9=image_previous[idx0+ppos9];float val_orig10=image_previous[idx0+ppos10];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p221(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		idx_type ppos1 =  patch_positions[1]; idx_type ppos2 =  patch_positions[2];
		idx_type ppos3 =  patch_positions[3]; idx_type ppos4 =  patch_positions[4];
		idx_type ppos5 =  patch_positions[5]; idx_type ppos6 =  patch_positions[6];
		idx_type ppos7 =  patch_positions[7]; idx_type ppos8 =  patch_positions[8];
		idx_type ppos9 =  patch_positions[9]; idx_type ppos10=  patch_positions[10];
		idx_type ppos11=  patch_positions[11]; idx_type ppos12=  patch_positions[12];
		idx_type ppos13 =  patch_positions[13]; idx_type ppos14=  patch_positions[14];

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0        ]; float val_orig1 = image_previous[idx0 + ppos1];
				float val_orig2 = image_previous[idx0 + ppos2]; float val_orig3 = image_previous[idx0 + ppos3];
				float val_orig4 = image_previous[idx0 + ppos4]; float val_orig5 = image_previous[idx0 + ppos5];
				float val_orig6 = image_previous[idx0 + ppos6]; float val_orig7 = image_previous[idx0 + ppos7];
				float val_orig8 = image_previous[idx0 + ppos8]; float val_orig9 = image_previous[idx0 + ppos9];
				float val_orig10= image_previous[idx0 + ppos10];float val_orig11= image_previous[idx0 + ppos11];
				float val_orig12= image_previous[idx0 + ppos12];float val_orig13= image_previous[idx0 + ppos13];
				float val_orig14= image_previous[idx0 + ppos14];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p222(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

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

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

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

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p331(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

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

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

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

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p332(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

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

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

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


				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p333(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

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

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

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

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
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
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}

	void IterativeNLM_CPU::filterslice_p111_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b];
				float val_orig1 = image_previous[idx0b + 1];
				float val_orig2 = image_previous[idx0b + 2];
				float val_orig3 = image_previous[idx0b + 3];
				float val_orig4 = image_previous[idx0b + 4];
				float val_orig5 = image_previous[idx0b + 5];
				float val_orig6 = image_previous[idx0b + 6];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b    ]-val_orig0; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.142857143f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.142857143f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p112_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b];
				float val_orig1 = image_previous[idx0b + 1];
				float val_orig2 = image_previous[idx0b + 2];
				float val_orig3 = image_previous[idx0b + 3];
				float val_orig4 = image_previous[idx0b + 4];
				float val_orig5 = image_previous[idx0b + 5];
				float val_orig6 = image_previous[idx0b + 6];
				float val_orig7 = image_previous[idx0b + 7];
				float val_orig8 = image_previous[idx0b + 8];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b    ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.04f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p113_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b];
				float val_orig1 = image_previous[idx0b + 1];float val_orig2 = image_previous[idx0b + 2];
				float val_orig3 = image_previous[idx0b + 3];float val_orig4 = image_previous[idx0b + 4];
				float val_orig5 = image_previous[idx0b + 5];float val_orig6 = image_previous[idx0b + 6];
				float val_orig7 = image_previous[idx0b + 7];float val_orig8 = image_previous[idx0b + 8];
				float val_orig9 = image_previous[idx0b + 9];float val_orig10= image_previous[idx0b +10];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.0204082f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p221_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b];
				float val_orig1 = image_previous[idx0b + 1];float val_orig2 = image_previous[idx0b + 2];
				float val_orig3 = image_previous[idx0b + 3];float val_orig4 = image_previous[idx0b + 4];
				float val_orig5 = image_previous[idx0b + 5];float val_orig6 = image_previous[idx0b + 6];
				float val_orig7 = image_previous[idx0b + 7];float val_orig8 = image_previous[idx0b + 8];
				float val_orig9 = image_previous[idx0b + 9];float val_orig10= image_previous[idx0b +10];
				float val_orig11= image_previous[idx0b +11];float val_orig12= image_previous[idx0b +12];
				float val_orig13= image_previous[idx0b +13];float val_orig14= image_previous[idx0b +14];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 11]-val_orig11; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 12]-val_orig12; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 13]-val_orig13; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 14]-val_orig14; distance += (tmp*tmp)*0.111111f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p222_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b        ]; float val_orig1 = image_previous[idx0b + 1];
				float val_orig2 = image_previous[idx0b + 2]; float val_orig3 = image_previous[idx0b + 3];
				float val_orig4 = image_previous[idx0b + 4]; float val_orig5 = image_previous[idx0b + 5];
				float val_orig6 = image_previous[idx0b + 6]; float val_orig7 = image_previous[idx0b + 7];
				float val_orig8 = image_previous[idx0b + 8]; float val_orig9 = image_previous[idx0b + 9];
				float val_orig10= image_previous[idx0b + 10];float val_orig11= image_previous[idx0b + 11];
				float val_orig12= image_previous[idx0b + 12];float val_orig13= image_previous[idx0b + 13];
				float val_orig14= image_previous[idx0b + 14];float val_orig15= image_previous[idx0b + 15];
				float val_orig16= image_previous[idx0b + 16];float val_orig17= image_previous[idx0b + 17];
				float val_orig18= image_previous[idx0b + 18];float val_orig19= image_previous[idx0b + 19];
				float val_orig20= image_previous[idx0b + 20];float val_orig21= image_previous[idx0b + 21];
				float val_orig22= image_previous[idx0b + 22];float val_orig23= image_previous[idx0b + 23];
				float val_orig24= image_previous[idx0b + 24];float val_orig25= image_previous[idx0b + 25];
				float val_orig26= image_previous[idx0b + 26];float val_orig27= image_previous[idx0b + 27];
				float val_orig28= image_previous[idx0b + 28];float val_orig29= image_previous[idx0b + 29];
				float val_orig30= image_previous[idx0b + 30];float val_orig31= image_previous[idx0b + 31];
				float val_orig32= image_previous[idx0b + 32];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 11]-val_orig11; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 12]-val_orig12; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 13]-val_orig13; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 14]-val_orig14; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 15]-val_orig15; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 16]-val_orig16; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 17]-val_orig17; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 18]-val_orig18; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 19]-val_orig19; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 20]-val_orig20; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 21]-val_orig21; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 22]-val_orig22; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 23]-val_orig23; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 24]-val_orig24; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 25]-val_orig25; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 26]-val_orig26; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 27]-val_orig27; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 28]-val_orig28; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 29]-val_orig29; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 30]-val_orig30; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 31]-val_orig31; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 32]-val_orig32; distance += (tmp*tmp)*0.04f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p331_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b        ]; float val_orig1 = image_previous[idx0b + 1];
				float val_orig2 = image_previous[idx0b + 2]; float val_orig3 = image_previous[idx0b + 3];
				float val_orig4 = image_previous[idx0b + 4]; float val_orig5 = image_previous[idx0b + 5];
				float val_orig6 = image_previous[idx0b + 6]; float val_orig7 = image_previous[idx0b + 7];
				float val_orig8 = image_previous[idx0b + 8]; float val_orig9 = image_previous[idx0b + 9];
				float val_orig10= image_previous[idx0b + 10];float val_orig11= image_previous[idx0b + 11];
				float val_orig12= image_previous[idx0b + 12];float val_orig13= image_previous[idx0b + 13];
				float val_orig14= image_previous[idx0b + 14];float val_orig15= image_previous[idx0b + 15];
				float val_orig16= image_previous[idx0b + 16];float val_orig17= image_previous[idx0b + 17];
				float val_orig18= image_previous[idx0b + 18];float val_orig19= image_previous[idx0b + 19];
				float val_orig20= image_previous[idx0b + 20];float val_orig21= image_previous[idx0b + 21];
				float val_orig22= image_previous[idx0b + 22];float val_orig23= image_previous[idx0b + 23];
				float val_orig24= image_previous[idx0b + 24];float val_orig25= image_previous[idx0b + 25];
				float val_orig26= image_previous[idx0b + 26];float val_orig27= image_previous[idx0b + 27];
				float val_orig28= image_previous[idx0b + 28];float val_orig29= image_previous[idx0b + 29];
				float val_orig30= image_previous[idx0b + 30];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 11]-val_orig11; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 12]-val_orig12; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 13]-val_orig13; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 14]-val_orig14; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 15]-val_orig15; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 16]-val_orig16; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 17]-val_orig17; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 18]-val_orig18; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 19]-val_orig19; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 20]-val_orig20; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 21]-val_orig21; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 22]-val_orig22; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 23]-val_orig23; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 24]-val_orig24; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 25]-val_orig25; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 26]-val_orig26; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 27]-val_orig27; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 28]-val_orig28; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 29]-val_orig29; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 30]-val_orig30; distance += (tmp*tmp)*0.111111f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p332_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b        ];float val_orig1=image_previous[idx0b+1];float val_orig2=image_previous[idx0b+2];float val_orig3=image_previous[idx0b+3];float val_orig4=image_previous[idx0b+4];
				float val_orig5=image_previous[idx0b+5];float val_orig6=image_previous[idx0b+6];float val_orig7=image_previous[idx0b+7];float val_orig8=image_previous[idx0b+8];
				float val_orig9=image_previous[idx0b+9];float val_orig10=image_previous[idx0b+10];float val_orig11=image_previous[idx0b+11];float val_orig12=image_previous[idx0b+12];
				float val_orig13=image_previous[idx0b+13];float val_orig14=image_previous[idx0b+14];float val_orig15=image_previous[idx0b+15];float val_orig16=image_previous[idx0b+16];
				float val_orig17=image_previous[idx0b+17];float val_orig18=image_previous[idx0b+18];float val_orig19=image_previous[idx0b+19];float val_orig20=image_previous[idx0b+20];
				float val_orig21=image_previous[idx0b+21];float val_orig22=image_previous[idx0b+22];float val_orig23=image_previous[idx0b+23];float val_orig24=image_previous[idx0b+24];
				float val_orig25=image_previous[idx0b+25];float val_orig26=image_previous[idx0b+26];float val_orig27=image_previous[idx0b+27];float val_orig28=image_previous[idx0b+28];
				float val_orig29=image_previous[idx0b+29];float val_orig30=image_previous[idx0b+30];float val_orig31=image_previous[idx0b+31];float val_orig32=image_previous[idx0b+32];
				float val_orig33=image_previous[idx0b+33];float val_orig34=image_previous[idx0b+34];float val_orig35=image_previous[idx0b+35];float val_orig36=image_previous[idx0b+36];
				float val_orig37=image_previous[idx0b+37];float val_orig38=image_previous[idx0b+38];float val_orig39=image_previous[idx0b+39];float val_orig40=image_previous[idx0b+40];
				float val_orig41=image_previous[idx0b+41];float val_orig42=image_previous[idx0b+42];float val_orig43=image_previous[idx0b+43];float val_orig44=image_previous[idx0b+44];
				float val_orig45=image_previous[idx0b+45];float val_orig46=image_previous[idx0b+46];float val_orig47=image_previous[idx0b+47];float val_orig48=image_previous[idx0b+48];
				float val_orig49=image_previous[idx0b+49];float val_orig50=image_previous[idx0b+50];float val_orig51=image_previous[idx0b+51];float val_orig52=image_previous[idx0b+52];
				float val_orig53=image_previous[idx0b+53];float val_orig54=image_previous[idx0b+54];float val_orig55=image_previous[idx0b+55];float val_orig56=image_previous[idx0b+56];
				float val_orig57=image_previous[idx0b+57];float val_orig58=image_previous[idx0b+58];float val_orig59=image_previous[idx0b+59];float val_orig60=image_previous[idx0b+60];
				float val_orig61=image_previous[idx0b+61];float val_orig62=image_previous[idx0b+62];float val_orig63=image_previous[idx0b+63];float val_orig64=image_previous[idx0b+64];
				float val_orig65=image_previous[idx0b+65];float val_orig66=image_previous[idx0b+66];float val_orig67=image_previous[idx0b+67];float val_orig68=image_previous[idx0b+68];
				float val_orig69=image_previous[idx0b+69];float val_orig70=image_previous[idx0b+70];float val_orig71=image_previous[idx0b+71];float val_orig72=image_previous[idx0b+72];


				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 11]-val_orig11; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 12]-val_orig12; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 13]-val_orig13; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 14]-val_orig14; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 15]-val_orig15; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 16]-val_orig16; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 17]-val_orig17; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 18]-val_orig18; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 19]-val_orig19; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 20]-val_orig20; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 21]-val_orig21; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 22]-val_orig22; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 23]-val_orig23; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 24]-val_orig24; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 25]-val_orig25; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 26]-val_orig26; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 27]-val_orig27; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 28]-val_orig28; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 29]-val_orig29; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 30]-val_orig30; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 31]-val_orig31; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 32]-val_orig32; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 33]-val_orig33; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 34]-val_orig34; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 35]-val_orig35; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 36]-val_orig36; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 37]-val_orig37; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 38]-val_orig38; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 39]-val_orig39; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 40]-val_orig40; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 41]-val_orig41; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 42]-val_orig42; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 43]-val_orig43; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 44]-val_orig44; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 45]-val_orig45; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 46]-val_orig46; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 47]-val_orig47; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 48]-val_orig48; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 49]-val_orig49; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 50]-val_orig50; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 51]-val_orig51; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 52]-val_orig52; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 53]-val_orig53; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 54]-val_orig54; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 55]-val_orig55; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 56]-val_orig56; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 57]-val_orig57; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 58]-val_orig58; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 59]-val_orig59; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 60]-val_orig60; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 61]-val_orig61; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 62]-val_orig62; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 63]-val_orig63; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 64]-val_orig64; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 65]-val_orig65; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 66]-val_orig66; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 67]-val_orig67; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 68]-val_orig68; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 69]-val_orig69; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 70]-val_orig70; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 71]-val_orig71; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 72]-val_orig72; distance += (tmp*tmp)*0.04f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_p333_unrolled(int z0, float multiplier, float* image_raw, float *image_previous, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		idx_type nslice = nx*ny;
		idx_type offset = (z0+zpad)*nslice;
		idx_type nslice_unpadded = shape[0]*shape[1];
		idx_type offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				idx_type idx0 = offset + y0*nx + x0;
				idx_type idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				float val_orig0 = image_previous[idx0b        ];float val_orig1=image_previous[idx0b+1];float val_orig2=image_previous[idx0b+2];float val_orig3=image_previous[idx0b+3];float val_orig4=image_previous[idx0b+4];
				float val_orig5=image_previous[idx0b+5];float val_orig6=image_previous[idx0b+6];float val_orig7=image_previous[idx0b+7];float val_orig8=image_previous[idx0b+8];
				float val_orig9=image_previous[idx0b+9];float val_orig10=image_previous[idx0b+10];float val_orig11=image_previous[idx0b+11];float val_orig12=image_previous[idx0b+12];
				float val_orig13=image_previous[idx0b+13];float val_orig14=image_previous[idx0b+14];float val_orig15=image_previous[idx0b+15];float val_orig16=image_previous[idx0b+16];
				float val_orig17=image_previous[idx0b+17];float val_orig18=image_previous[idx0b+18];float val_orig19=image_previous[idx0b+19];float val_orig20=image_previous[idx0b+20];
				float val_orig21=image_previous[idx0b+21];float val_orig22=image_previous[idx0b+22];float val_orig23=image_previous[idx0b+23];float val_orig24=image_previous[idx0b+24];
				float val_orig25=image_previous[idx0b+25];float val_orig26=image_previous[idx0b+26];float val_orig27=image_previous[idx0b+27];float val_orig28=image_previous[idx0b+28];
				float val_orig29=image_previous[idx0b+29];float val_orig30=image_previous[idx0b+30];float val_orig31=image_previous[idx0b+31];float val_orig32=image_previous[idx0b+32];
				float val_orig33=image_previous[idx0b+33];float val_orig34=image_previous[idx0b+34];float val_orig35=image_previous[idx0b+35];float val_orig36=image_previous[idx0b+36];
				float val_orig37=image_previous[idx0b+37];float val_orig38=image_previous[idx0b+38];float val_orig39=image_previous[idx0b+39];float val_orig40=image_previous[idx0b+40];
				float val_orig41=image_previous[idx0b+41];float val_orig42=image_previous[idx0b+42];float val_orig43=image_previous[idx0b+43];float val_orig44=image_previous[idx0b+44];
				float val_orig45=image_previous[idx0b+45];float val_orig46=image_previous[idx0b+46];float val_orig47=image_previous[idx0b+47];float val_orig48=image_previous[idx0b+48];
				float val_orig49=image_previous[idx0b+49];float val_orig50=image_previous[idx0b+50];float val_orig51=image_previous[idx0b+51];float val_orig52=image_previous[idx0b+52];
				float val_orig53=image_previous[idx0b+53];float val_orig54=image_previous[idx0b+54];float val_orig55=image_previous[idx0b+55];float val_orig56=image_previous[idx0b+56];
				float val_orig57=image_previous[idx0b+57];float val_orig58=image_previous[idx0b+58];float val_orig59=image_previous[idx0b+59];float val_orig60=image_previous[idx0b+60];
				float val_orig61=image_previous[idx0b+61];float val_orig62=image_previous[idx0b+62];float val_orig63=image_previous[idx0b+63];float val_orig64=image_previous[idx0b+64];
				float val_orig65=image_previous[idx0b+65];float val_orig66=image_previous[idx0b+66];float val_orig67=image_previous[idx0b+67];float val_orig68=image_previous[idx0b+68];
				float val_orig69=image_previous[idx0b+69];float val_orig70=image_previous[idx0b+70];float val_orig71=image_previous[idx0b+71];float val_orig72=image_previous[idx0b+72];
				float val_orig73=image_previous[idx0b+73];float val_orig74=image_previous[idx0b+74];float val_orig75=image_previous[idx0b+75];float val_orig76=image_previous[idx0b+76];
				float val_orig77=image_previous[idx0b+77];float val_orig78=image_previous[idx0b+78];float val_orig79=image_previous[idx0b+79];float val_orig80=image_previous[idx0b+80];
				float val_orig81=image_previous[idx0b+81];float val_orig82=image_previous[idx0b+82];float val_orig83=image_previous[idx0b+83];float val_orig84=image_previous[idx0b+84];
				float val_orig85=image_previous[idx0b+85];float val_orig86=image_previous[idx0b+86];float val_orig87=image_previous[idx0b+87];float val_orig88=image_previous[idx0b+88];
				float val_orig89=image_previous[idx0b+89];float val_orig90=image_previous[idx0b+90];float val_orig91=image_previous[idx0b+91];float val_orig92=image_previous[idx0b+92];
				float val_orig93=image_previous[idx0b+93];float val_orig94=image_previous[idx0b+94];float val_orig95=image_previous[idx0b+95];float val_orig96=image_previous[idx0b+96];
				float val_orig97=image_previous[idx0b+97];float val_orig98=image_previous[idx0b+98];float val_orig99=image_previous[idx0b+99];float val_orig100=image_previous[idx0b+100];
				float val_orig101=image_previous[idx0b+101];float val_orig102=image_previous[idx0b+102];float val_orig103=image_previous[idx0b+103];float val_orig104=image_previous[idx0b+104];
				float val_orig105=image_previous[idx0b+105];float val_orig106=image_previous[idx0b+106];float val_orig107=image_previous[idx0b+107];float val_orig108=image_previous[idx0b+108];
				float val_orig109=image_previous[idx0b+109];float val_orig110=image_previous[idx0b+110];float val_orig111=image_previous[idx0b+111];float val_orig112=image_previous[idx0b+112];
				float val_orig113=image_previous[idx0b+113];float val_orig114=image_previous[idx0b+114];

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					idx_type idx1 = idx0 + search_positions[s];
					idx_type idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					float tmp = 0.0f;
					tmp = image_previous[idx1b        ]-val_orig0; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 1]-val_orig1; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 2]-val_orig2; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 3]-val_orig3; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 4]-val_orig4; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 5]-val_orig5; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 6]-val_orig6; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 7]-val_orig7; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 8]-val_orig8; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 9]-val_orig9; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 10]-val_orig10; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 11]-val_orig11; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 12]-val_orig12; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 13]-val_orig13; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 14]-val_orig14; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 15]-val_orig15; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 16]-val_orig16; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 17]-val_orig17; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 18]-val_orig18; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 19]-val_orig19; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 20]-val_orig20; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 21]-val_orig21; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 22]-val_orig22; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 23]-val_orig23; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 24]-val_orig24; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 25]-val_orig25; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 26]-val_orig26; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 27]-val_orig27; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 28]-val_orig28; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 29]-val_orig29; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 30]-val_orig30; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 31]-val_orig31; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 32]-val_orig32; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 33]-val_orig33; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 34]-val_orig34; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 35]-val_orig35; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 36]-val_orig36; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 37]-val_orig37; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 38]-val_orig38; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 39]-val_orig39; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 40]-val_orig40; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 41]-val_orig41; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 42]-val_orig42; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 43]-val_orig43; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 44]-val_orig44; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 45]-val_orig45; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 46]-val_orig46; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 47]-val_orig47; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 48]-val_orig48; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 49]-val_orig49; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 50]-val_orig50; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 51]-val_orig51; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 52]-val_orig52; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 53]-val_orig53; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 54]-val_orig54; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 55]-val_orig55; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 56]-val_orig56; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 57]-val_orig57; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 58]-val_orig58; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 59]-val_orig59; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 60]-val_orig60; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 61]-val_orig61; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 62]-val_orig62; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 63]-val_orig63; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 64]-val_orig64; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 65]-val_orig65; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 66]-val_orig66; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 67]-val_orig67; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 68]-val_orig68; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 69]-val_orig69; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 70]-val_orig70; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 71]-val_orig71; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 72]-val_orig72; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 73]-val_orig73; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 74]-val_orig74; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 75]-val_orig75; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 76]-val_orig76; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 77]-val_orig77; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 78]-val_orig78; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 79]-val_orig79; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 80]-val_orig80; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 81]-val_orig81; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 82]-val_orig82; distance += (tmp*tmp)*0.111111f;
					tmp = image_previous[idx1b + 83]-val_orig83; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 84]-val_orig84; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 85]-val_orig85; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 86]-val_orig86; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 87]-val_orig87; distance += (tmp*tmp)*0.0682275f;
					tmp = image_previous[idx1b + 88]-val_orig88; distance += (tmp*tmp)*0.0501801f;
					tmp = image_previous[idx1b + 89]-val_orig89; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 90]-val_orig90; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 91]-val_orig91; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 92]-val_orig92; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 93]-val_orig93; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 94]-val_orig94; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 95]-val_orig95; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 96]-val_orig96; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 97]-val_orig97; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 98]-val_orig98; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 99]-val_orig99; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 100]-val_orig100; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 101]-val_orig101; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 102]-val_orig102; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 103]-val_orig103; distance += (tmp*tmp)*0.04f;
					tmp = image_previous[idx1b + 104]-val_orig104; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 105]-val_orig105; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 106]-val_orig106; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 107]-val_orig107; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 108]-val_orig108; distance += (tmp*tmp)*0.0333954f;
					tmp = image_previous[idx1b + 109]-val_orig109; distance += (tmp*tmp)*0.0287373f;
					tmp = image_previous[idx1b + 110]-val_orig110; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 111]-val_orig111; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 112]-val_orig112; distance += (tmp*tmp)*0.0225664f;
					tmp = image_previous[idx1b + 113]-val_orig113; distance += (tmp*tmp)*0.0204082f;
					tmp = image_previous[idx1b + 114]-val_orig114; distance += (tmp*tmp)*0.0204082f;
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance = distance*multiplier;
					//float this_weight = expf(distance); //primary time sink, using approximation instead
					float this_weight = (distance > expapproximation_cutoff) ? expapproximation(distance) : 0.0f;

					filtervalue += this_weight*noisy_value_searchpos;
					filterweight += this_weight;

					if (this_weight > maxweight) maxweight = this_weight;
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (maxweight > 0.0f)
				{
					filtervalue += maxweight*noisy_value_origin;
					filterweight += maxweight;

					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue/filterweight;
				}
				else
					result[offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin;

				//continue image space
			}
		}

		return;
	}
}


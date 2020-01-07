#include "iternlm_prepare.h"
#include <algorithm>

namespace denoise
{
	float* pad_reflective(float* imagestack, int padding[6], const int inshape[3], int outshape[3])
    {
        int nx0 = inshape[0]; int ny0 = inshape[1]; int nz0 = inshape[2];
        long long int nslice0 = nx0*ny0;

        outshape[0] = inshape[0] + padding[0] + padding[3];
        outshape[1] = inshape[1] + padding[1] + padding[4];
        outshape[2] = inshape[2] + padding[2] + padding[5];

        int nx1 = outshape[0]; int ny1 = outshape[1]; int nz1 = outshape[2];
        long long int nslice1 = nx1*ny1;
        long long int nstack1 = nslice1*nz1;

        float* output = (float*) malloc(nstack1*sizeof(*output));

		#pragma omp parallel for
		for (long long int idx1 = 0; idx1 < nstack1; idx1++)
		{
			int z1 = idx1/nslice1;
			int y1 = (idx1-z1*nslice1)/nx1;
			int x1 = idx1-z1*nslice1-y1*nx1;

			int z0 = z1-padding[2];
			int y0 = y1-padding[1];
			int x0 = x1-padding[0];

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

			output[idx1] = imagestack[idx0];
		}
        return output;
    }
	float* pad_reflective_unrollpatchspace(float* imagestack, int padding[6], const int inshape[3], int outshape[3], long long int *patchpositions, int nsize_patch)
	{
		int nx0 = inshape[0]; int ny0 = inshape[1]; int nz0 = inshape[2];
		long long int nslice0 = nx0*ny0;

		outshape[0] = inshape[0] + padding[0] + padding[3];
		outshape[1] = inshape[1] + padding[1] + padding[4];
		outshape[2] = inshape[2] + padding[2] + padding[5];

		int nx1 = outshape[0]; int ny1 = outshape[1]; int nz1 = outshape[2];
		long long int nslice1 = nx1*ny1;
		long long int nstack1 = nslice1*nz1;

		float* output = (float*) malloc((nstack1*nsize_patch)*sizeof(*output));

		#pragma omp parallel for
		for (long long int idx1 = 0; idx1 < nstack1; idx1++)
		{
			int z1 = idx1/nslice1;
			int y1 = (idx1-z1*nslice1)/nx1;
			int x1 = idx1-z1*nslice1-y1*nx1;

			for (int p = 0; p < nsize_patch; p++)
			{
				long long int patchshift = patchpositions[p];
				int zp = patchshift/nslice0;
				int yp = (patchshift-zp*nslice0)/nx0;
				int xp = patchshift-zp*nslice0-yp*nx0;

				int z0 = z1-padding[2]+zp;
				int y0 = y1-padding[1]+yp;
				int x0 = x1-padding[0]+xp;

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

				output[idx1*nsize_patch+p] = imagestack[idx0];
			}
		}

		return output;
	}

	long long int* setup_searchspace(int shape[3], protocol::DenoiseParameters *params, int &nsize_search)
	{
		//precalculate shifts in search space

		//image space
		int nx = shape[0]; int ny = shape[1];
		long long int nslice = nx*ny;

		//search space
		//////////////////////////////////////////////////////////////////////////////
		int nx_search = params->radius_searchspace[0]*2 + 1;
		int ny_search = params->radius_searchspace[1]*2 + 1;
		int nz_search = params->radius_searchspace[2]*2 + 1;
		long long int nslice_search = nx_search*ny_search;
		long long int nstack_search = nz_search*nslice_search;
		nsize_search = 0;

		float rxs = std::max(params->radius_searchspace[0], 1);
		float rys = std::max(params->radius_searchspace[1], 1);
		float rzs = std::max(params->radius_searchspace[2], 1);

		int nslices_searchspace = params->nslices;

		std::vector<long long int> searchidx;

		for (long long int idx_search = 0; idx_search < nstack_search; idx_search++)
		{
			int zs = idx_search/nslice_search;
			int ys = (idx_search-zs*nslice_search)/nx_search;
			int xs = idx_search-zs*nslice_search-ys*nx_search;

			zs -= params->radius_searchspace[2];
			ys -= params->radius_searchspace[1];
			xs -= params->radius_searchspace[0];

			if (abs(zs) > nslices_searchspace/2) continue; //search space out of bounds
			if ((zs == 0) && (ys == 0) && (xs == 0)) continue; //center will be weighted separately
			if (((xs/rxs)*(xs/rxs)+(ys/rys)*(ys/rys)+(zs/rzs)*(zs/rzs)) <= 1.f)
			{
				nsize_search++;
				searchidx.push_back((zs*nslice+ys*nx+xs));
			}
		}
		//////////////////////////////////////////////////////////////////////////////

		long long int *search_positions = (long long int*) malloc(nsize_search*sizeof(*search_positions));
		std::copy(searchidx.begin(), searchidx.end(), search_positions);

		return search_positions;
	}
	long long int* setup_patchspace(int shape[3], protocol::DenoiseParameters *params, int &nsize_patch)
	{
		//precalculate shifts in patch space

		//image space
		int nx = shape[0];
		long long int nslice = shape[0]*shape[1];

		//patch space
		//////////////////////////////////////////////////////////////////////////////
		int nx_patch = params->radius_patchspace[0]*2 + 1;
		int ny_patch = params->radius_patchspace[1]*2 + 1;
		int nz_patch = params->radius_patchspace[2]*2 + 1;
		long long int nslice_patch = nx_patch*ny_patch;
		long long int nstack_patch = nz_patch*nslice_patch;

		float rxp = std::max(params->radius_patchspace[0], 1);
		float ryp = std::max(params->radius_patchspace[1], 1);
		float rzp = std::max(params->radius_patchspace[2], 1);

		int nslices_patchspace = params->nslices;

		std::vector<long long int> patchidx;

		patchidx.push_back(0);
		nsize_patch = 1;

		for (long long int idx_patch = 0; idx_patch < nstack_patch; idx_patch++)
		{
			int zp = idx_patch/nslice_patch;
			int yp = (idx_patch-zp*nslice_patch)/nx_patch;
			int xp = idx_patch-zp*nslice_patch-yp*nx_patch;

			zp -= params->radius_patchspace[2];
			yp -= params->radius_patchspace[1];
			xp -= params->radius_patchspace[0];

			if (abs(zp) > nslices_patchspace/2) continue; //patch space out of bounds
			if ((zp == 0) && (yp == 0) && (xp == 0)) continue; //center will be weighted separately
			if (((xp/rxp)*(xp/rxp)+(yp/ryp)*(yp/ryp)+(zp/rzp)*(zp/rzp)) <= 1.f)
			{
				nsize_patch++;
				patchidx.push_back((zp*nslice+yp*nx+xp));
			}
		}
		//////////////////////////////////////////////////////////////////////////////

		long long int *patch_positions = (long long int*) malloc(nsize_patch*sizeof(*patch_positions));
		std::copy(patchidx.begin(), patchidx.end(), patch_positions);

		return patch_positions;
	}
	float* setup_distweight(int shape[3], protocol::DenoiseParameters *params)
	{
		//patch space
		//////////////////////////////////////////////////////////////////////////////
		int nx_patch = params->radius_patchspace[0]*2 + 1;
		int ny_patch = params->radius_patchspace[1]*2 + 1;
		int nz_patch = params->radius_patchspace[2]*2 + 1;
		long long int nslice_patch = nx_patch*ny_patch;
		long long int nstack_patch = nz_patch*nslice_patch;

		float rxp = std::max(params->radius_patchspace[0], 1);
		float ryp = std::max(params->radius_patchspace[1], 1);
		float rzp = std::max(params->radius_patchspace[2], 1);

		int nslices_patchspace = params->nslices;

		std::vector<float> distanceweight;
		distanceweight.push_back(1.f); //center will be reweighted
		int nsize_patch = 1;
		float maxweight = 0.0f;

		float sq_anisotropy = params->z_anisotropy * params->z_anisotropy;

		for (long long int idx_patch = 0; idx_patch < nstack_patch; idx_patch++)
		{
			int zp = idx_patch/nslice_patch;
			int yp = (idx_patch-zp*nslice_patch)/nx_patch;
			int xp = idx_patch-zp*nslice_patch-yp*nx_patch;

			zp -= params->radius_patchspace[2];
			yp -= params->radius_patchspace[1];
			xp -= params->radius_patchspace[0];

			if (abs(zp) > nslices_patchspace/2) continue; //patch space out of bounds
			if ((zp == 0) && (yp == 0) && (xp == 0)) continue; //center will be weighted separately
			if (((xp/rxp)*(xp/rxp)+(yp/ryp)*(yp/ryp)+(zp/rzp)*(zp/rzp)) <= 1.f)
			{
				//std::cout << xp << " " << yp << " " << zp << std::endl;
				float euclideandistance = sqrtf((xp*xp)+(yp*yp)+(zp*zp)*sq_anisotropy);
				float this_distance = 1.f/((2.f*euclideandistance+1.f)*(2.f*euclideandistance+1.f)); //apply distance function of choice

				if(this_distance > maxweight) maxweight = this_distance;

				distanceweight.push_back(this_distance);
				nsize_patch++;
			}
		}
		//////////////////////////////////////////////////////////////////////////////

		distanceweight[0] = maxweight;

		float *outdistanceweight = (float*) malloc(nsize_patch*sizeof(*outdistanceweight));

		if (rxp == 1.f && ryp == 1.f && rzp == 1.f && shape[2] > 1)
			for (int i = 0; i < nsize_patch; i++) outdistanceweight[i] = 1.f/7.f; //special case of radius 1
		else
			std::copy(distanceweight.begin(), distanceweight.end(), outdistanceweight);

		return outdistanceweight;
	}
	float* setup_gaussian_searchweight(float sigma, int shape[3], protocol::DenoiseParameters *params)
	{
		//additional weighting of search space

		//search space
		//////////////////////////////////////////////////////////////////////////////
		int nx_search = params->radius_searchspace[0]*2 + 1;
		int ny_search = params->radius_searchspace[1]*2 + 1;
		int nz_search = params->radius_searchspace[2]*2 + 1;
		long long int nslice_search = nx_search*ny_search;
		long long int nstack_search = nz_search*nslice_search;
		int nsize_search = 0;

		float rxs = std::max(params->radius_searchspace[0], 1);
		float rys = std::max(params->radius_searchspace[1], 1);
		float rzs = std::max(params->radius_searchspace[2], 1);

		int nslices_searchspace = params->nslices;

		std::vector<float> weights;

		for (long long int idx_search = 0; idx_search < nstack_search; idx_search++)
		{
			int zs = idx_search/nslice_search;
			int ys = (idx_search-zs*nslice_search)/nx_search;
			int xs = idx_search-zs*nslice_search-ys*nx_search;

			zs -= params->radius_searchspace[2];
			ys -= params->radius_searchspace[1];
			xs -= params->radius_searchspace[0];

			if (abs(zs) > nslices_searchspace/2) continue; //search space out of bounds
			if ((zs == 0) && (ys == 0) && (xs == 0)) continue; //center will be weighted separately
			if (((xs/rxs)*(xs/rxs)+(ys/rys)*(ys/rys)+(zs/rzs)*(zs/rzs)) <= 1.f)
			{
				nsize_search++;
				float dist = sqrtf(xs*xs+ys*ys+zs*zs);
				weights.push_back(expf((-dist*dist)/(2.f*sigma*sigma)));
			}
		}
		//////////////////////////////////////////////////////////////////////////////

		float *search_weights = (float*) malloc(nsize_search*sizeof(*search_weights));
		std::copy(weights.begin(), weights.end(), search_weights);

		return search_weights;
	}
}


#ifndef ITERNLM_PREPARE_H
#define ITERNLM_PREPARE_H

#include <vector>
#include <string.h>
#include <math.h>
#include <cstdint>
#include "auxiliary.h"
#include "hdcommunication.h"

namespace denoise
{
    class IterativeNLM_Preparation
    {
        public:
            std::pair<int,int> limited_rangedenoising = {-1, -1};
            //Get required files from disk:
            std::vector<float> get_substack(const std::string &path, const int &slice_nr, const int &nslices, const int patch_radius[3], int outshape[3]);
            std::vector<float> get_substack_limited(const std::string &path, const int &slice_nr, const int &nslices, const int patch_radius[3], int outshape[3]);
            //Pad first and second dimension with radius of search space using symmetric boundary conditions:
            std::vector<float> pad_substack(std::vector<float> &substack, int search_radius[3], const int inshape[3], int outshape[3]);
            //Copy the neighbours in the patch space after each voxel in the image
            std::vector<float> setup_patchspace(const std::vector<float> &substack, const int inshape[3], const uint_fast16_t &n_patch, const int patch_radius[3], int outshape[3]);
            //Now each patch in the search space can be accessed with a constant shift to the voxel in question. Here we precalculate these shifts:
            std::vector<int64_t>   setup_searchspace(const int radius[3],const int shape[3], const int &nslices, const int n_patch);
    };
}

#endif // ITERNLM_PREPARE_H


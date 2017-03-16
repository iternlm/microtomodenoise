#include "iternlm_prepare.h"

namespace denoise
{
    int enforceboundarycondition(int incoordinate, const int dimsize);

    std::vector<float> IterativeNLM_Preparation::get_substack(const std::string &path, const int &slice_nr, const int &nslices, const int patch_radius[3], int outshape[3])
    {
        std::vector<std::string> files;
        hdcom::HdCommunication hdcom;
        hdcom.GetFilelist(path, files);

        //Get the image and it's required neighbours and pad z-axis with patch_radius
        return hdcom.GetnSlices_32bit(slice_nr, nslices + 2*patch_radius[2], files, outshape);
    }
    std::vector<float> IterativeNLM_Preparation::get_substack_limited(const std::string &path, const int &slice_nr, const int &nslices, const int patch_radius[3], int outshape[3])
    {
        std::vector<std::string> files;
        hdcom::HdCommunication hdcom;
        hdcom.GetFilelist(path, files);

        if (limited_rangedenoising.second != -1) files.assign(files.begin(), files.begin()+limited_rangedenoising.second);
        if (limited_rangedenoising.first != -1) files.assign(files.begin()+limited_rangedenoising.first, files.end());

        //Get the image and it's required neighbours and pad z-axis with patch_radius
        return hdcom.GetnSlices_32bit(slice_nr, nslices + 2*patch_radius[2], files, outshape);
    }
    std::vector<float> IterativeNLM_Preparation::pad_substack(std::vector<float> &substack, int search_radius[3], const int inshape[3], int outshape[3])
    {
        int64_t firstpos = aux::xy2idx(-search_radius[0], -search_radius[1], inshape);
        int64_t lastpos = substack.size()-firstpos;

        std::vector<float> outstack;
        outstack.reserve(lastpos-firstpos);

        //Iterate over the padded vector
        int this_h, this_w;
        int64_t pos;

        for (int d = 0; d < inshape[2]; d++)
        {
            for (int h = -search_radius[1]; h < inshape[1] + search_radius[1]; h++)
            {
                //Enforce a symmetric boundary
                this_h = enforceboundarycondition(h, inshape[1]);

                for (int w = -search_radius[0]; w < inshape[0] + search_radius[0]; w++)
                {
                    this_w = enforceboundarycondition(w, inshape[0]);
                    pos = aux::xyz2idx(this_w, this_h, d, inshape);

                    outstack.push_back(substack[pos]);
                }
            }
        }

        outshape[0] = inshape[0] + 2*search_radius[0];
        outshape[1] = inshape[1] + 2*search_radius[1];
        outshape[2] = inshape[2];
        return outstack;
    }
    std::vector<float> IterativeNLM_Preparation::setup_patchspace(const std::vector<float> &substack, const int inshape[3], const uint_fast16_t &n_patch, const int patch_radius[3], int outshape[3])
    {
        double a,b,c;

        a = patch_radius[0];
        b = patch_radius[1];
        c = patch_radius[2];

        std::vector<float> outstack;
        outstack.reserve(n_patch*substack.size());

        int length = substack.size();
        int x,y,z,x_patch,y_patch,z_patch;

        for(int64_t pos=0; pos < length; pos++)
        {
            //Central voxel is always in first position
            outstack.push_back(substack[pos]);
            aux::idx2xyz(pos, inshape, x, y, z);

            for (int d = -patch_radius[2]; d <= patch_radius[2]; d++)
            {
                z_patch = enforceboundarycondition(z+d, inshape[2]);

                for (int w = -patch_radius[1]; w <= patch_radius[1]; w++)
                {
                    y_patch = enforceboundarycondition(y+w, inshape[1]);

                    for (int h = -patch_radius[0]; h <= patch_radius[0]; h++)
                    {
                        if((h == 0) && (w == 0) && (d == 0))
                            continue;

                        if (((h/a)*(h/a)+(w/b)*(w/b)+(d/c)*(d/c)) <= 1.)
                        {
                            x_patch = enforceboundarycondition(x+h, inshape[0]);
                            outstack.push_back(substack[aux::xyz2idx(x_patch, y_patch, z_patch, inshape)]);
                        }
                    }
                }
            }
        }

        outshape[0] = n_patch*inshape[0];
        outshape[1] = inshape[1];
        outshape[2] = inshape[2];

        return outstack;
    }
    std::vector<int64_t> IterativeNLM_Preparation::setup_searchspace(const int radius[3],const int shape[3], const int &nslices, const int n_patch)
    {
        double a,b,c;
        int64_t idx;
        std::vector<int64_t> search_positions;

        a = radius[0];
        b = radius[1];
        c = radius[2];

        for (int d = -radius[2]; d <= radius[2]; d++)
        {
            if ((d > (nslices/2)) || (-d > (nslices/2)))
            {
                //search space out of bounds
                continue;
            }

            for (int w = -radius[1]; w <= radius[1]; w++)
            {
                for (int h = -radius[0]; h <= radius[0]; h++)
                {
                    if ((h == 0) && (w == 0) && (d == 0))
                    {
                        //Center will be weighted separately
                        continue;
                    }

                    if (((h/a)*(h/a)+(w/b)*(w/b)+(d/c)*(d/c)) <= 1.)
                    {
                        idx = aux::xyz2idx(h*n_patch, w, d, shape);
                        search_positions.push_back(idx);
                    }
                }
            }
        }
        return search_positions;
    }

    /*************** Helper Functions ****************/
    int enforceboundarycondition(int incoordinate, const int dimsize)
    {
        if (incoordinate < 0)
            return -incoordinate;
        else if (incoordinate >= dimsize)
            return 2*dimsize-incoordinate-2;
        return incoordinate;
    }
}


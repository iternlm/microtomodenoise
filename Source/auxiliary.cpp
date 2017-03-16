#include <iostream>
#include <vector>
#include <algorithm>

namespace aux
{
    /*2D-Conversions
    *********************************************************/
    void idx2xy(const int64_t &pos, const int shape[2], int &outx, int &outy)
    {
        outy = pos/shape[0];
        outx = pos-(outy*shape[0]);
        return;
    }
    int64_t xy2idx(const int &x,const int &y, const int shape[2])
    {
        return y*shape[0]+x;
    }

    /*3D-Conversions
    *********************************************************/
    void idx2xyz(const int64_t &pos, const int shape[3], int &outx, int &outy, int &outz)
    {
        outz = pos/(shape[0]*shape[1]);
        int remainder = pos-(outz*shape[0]*shape[1]);
        outy = remainder/shape[0];
        outx = remainder-(outy*shape[0]);
        return;
    }
    int64_t xyz2idx(const int &x, const int &y, const int &z, const int shape[3])
    {
        return x+y*shape[0]+z*shape[0]*shape[1];
    }

    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill)
    {
        std::string outstring = std::to_string(inint);
        while(outstring.length() < zfill)
            outstring = "0" + outstring;
        return outstring;
    }

    /*Numpy-like
    *********************************************************/
    std::vector<double> linspace(double startval, double endval, uint64_t bins)
    {
        std::vector<double> linspaced(bins);
        double delta = (endval-startval)/(bins-1);
        for(uint64_t i = 0; i < (bins-1); i++)
        {
            linspaced[i] = startval + delta * i;
        }
        linspaced[bins-1] = endval;
        return linspaced;
    }
    void rescale(std::vector<float> &imgdata, float lb, float ub, float minval, float maxval)
    {
        for(uint64_t idx = 0; idx < imgdata.size(); idx++)
        {
            float this_value = imgdata[idx];

            if (this_value <= minval)
                imgdata[idx] = lb;
            else if (this_value >= maxval)
                imgdata[idx] = ub;
            else
                imgdata[idx] = (ub-lb)*(this_value-minval)/(maxval-minval)+lb;
        }
        return;
    }
    void rescale(std::vector<float> &imgdata, float lb, float ub, float minval, float maxval, bool cutoffoob)
    {
        for(uint64_t idx = 0; idx < imgdata.size(); idx++)
        {
            float this_value = imgdata[idx];

            if(cutoffoob == true)
            {
                if (this_value <= minval)
                    imgdata[idx] = lb;
                else if (this_value >= maxval)
                    imgdata[idx] = ub;
                else
                    imgdata[idx] = (ub-lb)*(this_value-minval)/(maxval-minval)+lb;
            }
            else
            {
                imgdata[idx] = (ub-lb)*(this_value-minval)/(maxval-minval)+lb;
            }
        }
        return;
    }
    void rescale(std::vector<float> &imgdata, float lb, float ub)
    {
        float minval = *std::min_element(imgdata.begin(), imgdata.end());
        float maxval = *std::max_element(imgdata.begin(), imgdata.end());

        for(uint64_t idx = 0; idx < imgdata.size(); idx++)
        {
            imgdata[idx] = (ub-lb)*(imgdata[idx]-minval)/(maxval-minval)+lb;
        }
        return;
    }
    std::vector<size_t> argsort_ascending(const std::vector<uint64_t> &v)
    {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota( idx.begin(), idx.end(), 0 );

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2)
        {
            return v[i1] < v[i2];
        });

        return idx;
    }
    std::vector<size_t> argsort_descending(const std::vector<uint64_t> &v)
    {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota( idx.begin(), idx.end(), 0 );

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2)
        {
            return v[i1] > v[i2];
        });

        return idx;
    }
}


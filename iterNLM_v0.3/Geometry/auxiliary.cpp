#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace aux
{
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
}

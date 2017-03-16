/*
 ************************************************************************************
 *
 * This file provides basic functionality needed for 3D iterative nonlocal means denoising
 * of spatially invariant noise (Gaussian noise) as described in:
 *
 * S. Bruns, H. Osholm Sørensen, S.L.S. Stipp
 *
 * "Looking for the Signal: A Guide to Iterative Noise and Artefact Removal in X-ray
 * Tomography Reconstructions of Reservoir Rocks"
 *
 ************************************************************************************
 *
 * How to use:
 *
 * After initializing an IterativeNLM-object the user needs to provide a std::string specifying
 * a path to an image-sequence in tif-format, the noise standard deviation ('sigma') and the
 * current iteration count starting at 1.
 *
 * 'Run_GaussianNoise_Stack' executes one iteration of denoising for the entire image sequence.
 * 'Run_GaussianNoise_SingleSlice' executes denoising on a single image file identified by it's position in the image sequence.
 *
 * Some of the default settings for denoising can be altered after initialization via the set-functions.
 *
 ************************************************************************************
 *
 * Notes:
 *
 * - It is assumed that IO-operations are not a major bottleneck but the calculation of the exp during filtering.
 *   Thus, 'Run_GaussianNoise_SingleSlice' reads all required slices from disk with redundancy.
 *
 * - iternlm_prepare.h provides the functionality to acquire the data and put them in a shape that is efficiently filtered.
 *   The implementation of the actual filtering is part of this class/file to avoid losses in performance.
 *
 * - Image data are treated as 1D-vector. Navigation is provided with the Auxiliary-class.
 *
 * - For compilation link 'libtiff', 'boost_filesystem', 'boost_system', 'gomp' and compile with -fopenmp and C++0x ISO C-standard.
 *
 ************************************************************************************
 */

#ifndef ITERNLM_H
#define ITERNLM_H

#include <iostream>
#include <vector>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cstdint>
#include "auxiliary.h"
#include "hdcommunication.h"
#include "iternlm_prepare.h"

namespace denoise
{
    class IterativeNLM
    {
        /************ Default Parameter Settings ************/
    public:
        int nslices = 3; //Number of adjacent slices used in denoising.
        int patch_radius[3] = {1, 1, 1}; //Radius of spherical patch space by axis.
        int search_radius[3] = {10, 10, 2}; //Radius of ellipsoidal search space.
        float beta = 1.; //allows to change the smoothing parameter

        bool resume = false; //skip files that have already been denoised.
        bool approximate = true; //the exponential is the bottleneck. Set to true if expapproximation-function should be used instead.
        bool output_16bit = false;
        std::pair<int,int> limited_rangedenoising = {-1, -1};
        /************ Extended Functionality ************/

    private:
        float distancefunction(float x) {return 1./((2*x+1)*(2*x+1));}   //irrelevant when patch_radius is 1

        //3rd order Padé approximation:
        float expapproximation(float x){return (1+.5*x+1.11111111e-1*x*x+1.38888889e-2*x*x*x)/(1-.5*x+1.11111111e-1*x*x-1.38888889e-2*x*x*x);}
        float expapproximation_cutoff = -3.56648; //inform the compiler when the expapproximation becomes negative

        /*//5thorder Padé approximation:
        float expapproximation(float x){float x2 = x*x; return (1+.5*x+.111111111*x2+.013888889*x2*x+.000992063*x2*x2+.000033069*x2*x2*x)
                                                              /(1-.5*x+.111111111*x2-.013888889*x2*x+.000992063*x2*x2-.000033069*x2*x2*x);}*/
        /*Polynomial approximation: float expapproximation(float x) {return 1.00043+x*(1.00946+x*(0.50633+x*(0.15793+x*(0.03117+x*(0.00317)))));}*/

        /************ User functions ************/
    public:
        //constructor and parameter access
        IterativeNLM() {}

        //Execution
        void Run_GaussianNoise_Stack(const std::string &path_noisy, const std::string &path_result, std::vector<double> &sigma, const int &iteration, int &outnslices);
        void Run_GaussianNoise_SingleSlice(const std::string &path_noisy, const std::string &path_result, const int &slice_nr, const float &sigma, const int &iteration);

        /************ Helper functions ************/
    private:
        std::vector<float> getdistweight(uint_fast16_t &outn_patch); //calculates a kernel of weights (according to the distancefunction) for patch radii > 1

        //The NLM-filtering procedure for one slice:
        std::vector<float> filterslice(const float &divisor,const int64_t &firstpos, const int64_t &lastpos, const int inshape[3],
                                       std::vector<float> &activeslices, std::vector<float> &prefiltered_slices, std::vector<int64_t> &search_positions, std::vector<float> &distweight);
        std::vector<float> filterslice_approximation(const float &divisor,const int64_t &firstpos, const int64_t &lastpos, const int inshape[3],
                                       std::vector<float> &activeslices, std::vector<float> &prefiltered_slices, std::vector<int64_t> &search_positions, std::vector<float> &distweight);
        float calculatedistance(const std::vector<float>::iterator &pos1, const std::vector<float>::iterator &pos2, const uint_fast16_t &n_patch, const std::vector<float> &distweight);
        /****************************************/
    };
}
#endif // ITERNLM_H


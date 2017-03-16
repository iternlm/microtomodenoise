#include "iternlm.h"

namespace denoise
{
    /********************* Execution *********************/
    void IterativeNLM::Run_GaussianNoise_Stack(const std::string &path_noisy, const std::string &path_result, std::vector<double> &sigma, const int &iteration, int &outnslices)
    {
        hdcom::HdCommunication hdcom;
        int n_slices = hdcom.GetFileAmount(path_noisy);
        if (limited_rangedenoising.second != -1) n_slices = std::min(n_slices, limited_rangedenoising.second);
        if (limited_rangedenoising.first != -1) n_slices -= limited_rangedenoising.first;
        else limited_rangedenoising.first = 0;

        outnslices = n_slices; //poor choice of variable as its overlapping a bit with global variable
        if (n_slices < 1) {
            std::cout << "No slices to denoise!" << std::endl;
            return;
        }
        if (n_slices == 1) {
            std::cout << "2D denoising is not supported in this version" << std::endl;
            return;
        }
        if (n_slices < (nslices/2+1))
        {
            //reduce the amount of slices for short image sequences
            if (n_slices <= 3) nslices = 1;
            else nslices = n_slices/2+1;
        }

        std::string outpath = path_result+"/denoise-iteration"+aux::zfill_int2string(iteration,2)+"/";
        hdcom.makedir(outpath);

        #pragma omp parallel //num_threads(8)
        {
            #pragma omp for
            for (int i = 0; i < n_slices; i++)
            {
                //Resume if slice has already been processed
                if (resume)
                {
                    if (boost::filesystem::exists( outpath + "denoised" + aux::zfill_int2string(i + limited_rangedenoising.first, 4)+".tif" ) )
                    {
                        //std::cout << "Slice " << i << " already exists!" << std::endl;
                        continue;
                    }
                }

                if (i < sigma.size()) Run_GaussianNoise_SingleSlice(path_noisy, path_result, i, sigma[i], iteration);
                else Run_GaussianNoise_SingleSlice(path_noisy, path_result, i, sigma[0], iteration);

                std::cout << "\riteration " << iteration << " slice " << i << " finished!";
                std::cout.flush();
            }
        }

        return;
    }

    void IterativeNLM::Run_GaussianNoise_SingleSlice(const std::string &path_noisy, const std::string &path_result, const int &slice_nr, const float &sigma, const int &iteration)
    {
        using namespace std;
        IterativeNLM_Preparation prep; //New object of data preparation class
        prep.limited_rangedenoising.first = limited_rangedenoising.first;
        prep.limited_rangedenoising.second = limited_rangedenoising.second;

        uint_fast16_t n_patch; //amount of voxels in patch space

        /********** Parameters controlling the denoising result *********************/
        float divisor = sigma*sigma*beta; //changes depending on way of implementation. Beta is available if control needed
        std::vector<float> distweight = getdistweight(n_patch);
        /***************************************************************************/

        /******************* Acquisition of noisy data *****************************/
        int shape_input[3];    //remembers the original shape of the substack
        int shape_padded[3]; //remembers the shape of the padded data
        int shape_full[3];  //remembers the shape of the patch expanded data we are looping over
        vector<float> activeslices = prep.get_substack_limited(path_noisy, slice_nr, nslices, patch_radius, shape_input);
        activeslices = prep.pad_substack(activeslices,search_radius, shape_input, shape_padded); //pad the acquired data with the search_radius
        activeslices = prep.setup_patchspace(activeslices, shape_padded, n_patch, patch_radius, shape_full); //puts the neighbourhood of a voxel in subsequent locations of the vector bloating it's size
        /***************************************************************************/

        /******************* Acquisition of prefiltered data ***********************/
        vector<float> prefiltered_slices;
        if (iteration == 1)
            prefiltered_slices = activeslices;
        else
        {
            string path_prefiltered = path_result + "/denoise-iteration" + aux::zfill_int2string(iteration-1, 2) + "/";
            prefiltered_slices = prep.get_substack(path_prefiltered, slice_nr, nslices, patch_radius, shape_input);
            prefiltered_slices = prep.pad_substack(prefiltered_slices, search_radius, shape_input, shape_padded);
            prefiltered_slices = prep.setup_patchspace(prefiltered_slices, shape_padded, n_patch, patch_radius, shape_full);
        }
        /*****************************************************************************/

        /** Identify ROI with padding (no need to process outside these boundaries) **/
        int64_t firstpos = (activeslices.size() - (shape_full[0]*shape_full[1])) / 2;
        int64_t lastpos = firstpos + (shape_full[0]*shape_full[1]);
        //shift start and end to exclude padding
        firstpos += search_radius[1]*shape_full[0] + search_radius[0]*n_patch;
        lastpos -= search_radius[1]*shape_full[0];
        /*****************************************************************************/

        //Now set up the search space as shifts in the padded and expanded stack.
        //This is already sorted in ascending order and must only be performed once
        vector<int64_t> search_positions = prep.setup_searchspace(search_radius, shape_full, nslices, n_patch);

        /****************   Start the filtering process ******************************/
        vector<float> outslice;
        if (!approximate)
            outslice = filterslice(divisor, firstpos, lastpos, shape_input, activeslices, prefiltered_slices, search_positions, distweight);
        else
            outslice = filterslice_approximation(divisor, firstpos, lastpos, shape_input, activeslices, prefiltered_slices, search_positions, distweight);

        /*****************************************************************************/

        /**************** Write the filtered data to the disk *************************/
        hdcom::HdCommunication hdcom;
        string outpath = path_result + "/denoise-iteration" + aux::zfill_int2string(iteration, 2) + "/";
        string filename = "denoised" + aux::zfill_int2string(slice_nr+limited_rangedenoising.first, 4);

        if (output_16bit)
        {
            vector<uint16_t> outslice16(outslice.size(), 0);

            //quick and dirty workaround to get a version with 16bit output. Just convert.
            for (uint64_t idx = 0; idx < outslice.size(); idx++)
                outslice16[idx] = min(max(0., (double) round(outslice[idx])), 65535.);

            hdcom.Save2DTifImage_16bit(outslice16, shape_input, outpath, filename);
        }
        else
            hdcom.Save2DTifImage_32bit(outslice, shape_input, outpath, filename);
        /*****************************************************************************/

        return;
    }

    /**************** Filtering Functions *****************/
    std::vector<float> IterativeNLM::filterslice(const float &divisor, const int64_t &firstpos, const int64_t &lastpos, const int inshape[3],
            std::vector<float> &activeslices, std::vector<float> &prefiltered_slices, std::vector<int64_t> &search_positions, std::vector<float> &distweight)
    {
        /*
         * The actual filtering takes place in here. No further refactoring because function calls within the iterator kill the performance!
         */
        using namespace std;

        uint_fast16_t n_searchspace = search_positions.size(); //amount of voxels in the search space
        uint_fast16_t n_patch = distweight.size(); //size of patch space

        vector<float> outslice(inshape[0]*inshape[1],0.); //This will be the filtered image
        vector<float>::iterator pos2; //points to the patch the active voxel is compared to
        float distance, this_weight, filtervalue, filterweight, maxweight;
        uint64_t outpos = 0; //position in the denoised image
        uint64_t this_pos = firstpos; //remembers the location in the noisy data

        for(vector<float>::iterator pos1 = prefiltered_slices.begin()+firstpos; pos1 < prefiltered_slices.begin()+lastpos; pos1+=n_patch)
        {
            filtervalue = 0;
            filterweight = 0;
            maxweight = 0;

            for (uint_fast16_t i = 0; i < n_searchspace; i++)
            {
                pos2 = pos1+search_positions[i];

                distance = calculatedistance(pos1, pos2, n_patch, distweight);

                this_weight = exp(-distance/divisor); //primary timesink

                pos2 = activeslices.begin()+search_positions[i]+this_pos; //coupling to noisy data by weighting the original values

                filtervalue += this_weight*(*pos2);
                filterweight += this_weight;

                if (this_weight > maxweight)
                    maxweight = this_weight;
            }

            //reweight the center
            pos2 = activeslices.begin()+this_pos;

            if (maxweight > 0)
            {
                filtervalue += maxweight*(*pos2);
                filterweight += maxweight;

                //write filtered value to output
                outslice[outpos] = filtervalue/filterweight;
            }
            else
                outslice[outpos] = *pos2;

            outpos++;

            //Identify when a line is calculated and fast forward
            if (outpos%inshape[0] == 0)
            {
                pos1 += 2*search_radius[0]*n_patch;
                this_pos += 2*search_radius[0]*n_patch;
            }

            this_pos += n_patch;
        }
        return outslice;
    }
    std::vector<float> IterativeNLM::filterslice_approximation(const float &divisor, const int64_t &firstpos, const int64_t &lastpos, const int inshape[3],
            std::vector<float> &activeslices, std::vector<float> &prefiltered_slices, std::vector<int64_t> &search_positions, std::vector<float> &distweight)
    {
        /*
         * The actual filtering takes place in here. No further refactoring because function calls within the iterator kill the performance!
         */
        using namespace std;

        uint_fast16_t n_searchspace = search_positions.size(); //amount of voxels in the search space
        uint_fast16_t n_patch = distweight.size(); //size of patch space

        vector<float> outslice(inshape[0]*inshape[1],0.); //This will be the filtered image
        vector<float>::iterator pos2; //points to the patch the active voxel is compared to
        float distance, this_weight, filtervalue, filterweight, maxweight;
        uint64_t outpos = 0; //position in the denoised image
        int64_t this_pos = firstpos; //remembers the location in the noisy data

        for(vector<float>::iterator pos1 = prefiltered_slices.begin()+firstpos; pos1 < prefiltered_slices.begin()+lastpos; pos1+=n_patch)
        {
            filtervalue = 0;
            filterweight = 0;
            maxweight = 0;

            for (uint_fast16_t i = 0; i < n_searchspace; i++)
            {
                pos2 = pos1+search_positions[i];

                distance = calculatedistance(pos1, pos2, n_patch, distweight);
                distance = -distance/divisor;

                if (distance > expapproximation_cutoff)
                    this_weight = expapproximation(distance);
                else
                    this_weight = 0;

                pos2 = activeslices.begin()+search_positions[i]+this_pos; //coupling to noisy data by weighting the original values

                filtervalue += this_weight*(*pos2);
                filterweight += this_weight;

                if (this_weight > maxweight)
                    maxweight = this_weight;
            }

            //reweight the center
            pos2 = activeslices.begin()+this_pos;

            if (maxweight > 0)
            {
                filtervalue += maxweight*(*pos2);
                filterweight += maxweight;

                //write filtered value to output
                outslice[outpos] = filtervalue/filterweight;
            }
            else
                outslice[outpos] = *pos2;

            outpos++;

            //Identify when a line is calculated and fast forward
            if (outpos%inshape[0] == 0)
            {
                pos1 += 2*search_radius[0]*n_patch;
                this_pos += 2*search_radius[0]*n_patch;
            }

            this_pos += n_patch;
        }

        return outslice;
    }

    float IterativeNLM::calculatedistance(const std::vector<float>::iterator &pos1, const std::vector<float>::iterator &pos2,
                                          const uint_fast16_t &n_patch, const std::vector<float> &distweight)
    {
        float tmp = *pos1-*pos2;
        float distance = tmp*tmp*distweight[0];

        for (uint_fast16_t i = 1; i < n_patch; i++)
        {
            tmp = *(pos1+i)-*(pos2+i);
            distance += tmp*tmp*distweight[i];
        }
        return distance;
    }
    /******************************************************/

    /****************** Helper Functions ******************/
    std::vector<float> IterativeNLM::getdistweight(uint_fast16_t &outn_patch)
    {
        outn_patch = 1;
        float a,b,c, euclideandistance, this_weight;
        double maxweight = 0;

        a = patch_radius[0];
        b = patch_radius[1];
        c = patch_radius[2];

        std::vector<float> distweights;

        //Early exit for patch radius = 1
        if ((patch_radius[0] == 1) && (patch_radius[1] == 1) && (patch_radius[2] == 1))
        {
            outn_patch = 7;
            distweights = {1./7. ,1./7., 1./7., 1./7., 1./7., 1./7., 1./7.};
            return distweights;
        }

        distweights.push_back(1.); //Center will be reweighted

        for (int d = -patch_radius[2]; d <= patch_radius[2]; d++)
        {
            for (int w = -patch_radius[1]; w <= patch_radius[1]; w++)
            {
                for (int h = -patch_radius[0]; h <= patch_radius[0]; h++)
                {

                    if ((h == 0) && (w == 0) && (d == 0))
                        continue;
                    if (((h/a)*(h/a)+(w/b)*(w/b)+(d/c)*(d/c)) <= 1.)
                    {
                        outn_patch++;
                        euclideandistance = sqrt((d*d)+(w*w)+(h*h));
                        this_weight = distancefunction(euclideandistance);

                        if (this_weight > maxweight)
                            maxweight = this_weight;

                        distweights.push_back(this_weight);
                    }
                }
            }
        }
        distweights[0] = maxweight;
        return distweights;
    }

    /******************************************************/
}


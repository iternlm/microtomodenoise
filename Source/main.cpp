#include <iostream>
#include <ctime>
#include "hdcommunication.h"
#include "iternlm.h"

using namespace std;

int main(int argc, char* argv[])
{
    std::string path_rawdata = "/";
    std::string path_resdata, rootpath;

    double sigma0 = 0.2;
    double sigma1 = 0.06;

    ////// Set the denoising parameters //////
    denoise::IterativeNLM nlm;

    double alpha = 0.5;
    int maxiterations = 4;

    nlm.resume = true;
    nlm.nslices = 11;
    nlm.search_radius[0] = 10;
    nlm.search_radius[1] = 10;
    nlm.search_radius[2] = 5;

    bool output_16bit = false;

    //limited range
    int firstslice = -1;
    int lastslice = -1;

    if ("extract comman line arguments)"){
        for (uint16_t i = 1; i < argc; i++)
        {
            if ((string(argv[i]) == "-i") || (string(argv[i]) == "-input"))
            {
                i++;
                path_rawdata = string(argv[i]);
                if (path_rawdata.substr(path_rawdata.length()-1, 1) != "/")
                    path_rawdata.append("/");
            }
            else if((string(argv[i]) == "-o") || (string(argv[i]) == "-output"))
            {
                i++;
                path_resdata = string(argv[i]);
                if (path_resdata.substr(path_resdata.length()-1, 1) != "/")
                    path_resdata.append("/");
            }
            else if ((string(argv[i]) == "-s0") || (string(argv[i]) == "-sigma0"))
            {
                i++;
                sigma0 = atof(argv[i]);
            }
            else if ((string(argv[i]) == "-s1") || (string(argv[i]) == "-sigma1"))
            {
                i++;
                sigma1 = atof(argv[i]);
            }
            else if ((string(argv[i]) == "-s") || (string(argv[i]) == "-sigma"))
            {
                i++;
                sigma0 = atof(argv[i]);
                sigma1 = atof(argv[i]);
            }
            else if ((string(argv[i]) == "-fs") || (string(argv[i]) == "-firstlice"))
            {
                i++;
                firstslice = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-ls") || (string(argv[i]) == "-lastslice"))
            {
                i++;
                lastslice = atoi(argv[i])+1; //adding 1 because lastslice is exclusive
            }
            else if ((string(argv[i]) == "-r") || (string(argv[i]) == "-resume"))
            {
                i++;
                if ((string(argv[i]) == "false") || (string(argv[i]) == "False"))
                    nlm.resume = false;
                else if ((string(argv[i]) == "true") || (string(argv[i]) == "True"))
                    nlm.resume = true;
            }
            else if ((string(argv[i]) == "-iter") || (string(argv[i]) == "-iterations"))
            {
                i++;
                maxiterations = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-slices") || (string(argv[i]) == "-nslices"))
            {
                i++;
                nlm.nslices = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-search0"))
            {
                i++;
                nlm.search_radius[0] = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-search1"))
            {
                i++;
                nlm.search_radius[1] = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-search2"))
            {
                i++;
                nlm.search_radius[2] = atoi(argv[i]);
            }
            else if ((string(argv[i]) == "-a") || (string(argv[i]) == "-alpha"))
            {
                i++;
                alpha = atof(argv[i]);
            }
            else if ((string(argv[i]) == "--16bit"))
                output_16bit = true;
            else if (path_rawdata.length() == 0)
                path_rawdata = string(argv[i]);
            else if (path_resdata.length() == 0)
                path_resdata = string(argv[i]);
        }
        rootpath = path_rawdata.substr(0, path_rawdata.rfind("/", path_rawdata.length()-2)+1);
        if (path_resdata.length() == 0){
            rootpath = path_rawdata.substr(0, path_rawdata.rfind("/", path_rawdata.length()-2)+1);
            path_resdata = rootpath + "/denoised/";}
        else
            rootpath = path_resdata.substr(0, path_resdata.rfind("/", path_resdata.length()-2)+1);
    }

    if (output_16bit) nlm.output_16bit = true;
    nlm.limited_rangedenoising.first = firstslice;
    nlm.limited_rangedenoising.second = lastslice;

    ////// Run the first denoising iteration //////
    int slices;
    std::vector<double> sigma{sigma0};
    nlm.Run_GaussianNoise_Stack(path_rawdata, path_resdata, sigma, 1, slices);

    ////// Finish subsequent denoising iterations //////
    int iteration = 2;

    clock_t startTime, endTime, clockTicksTaken;
    double timeInSeconds;
    int performed_iters = maxiterations-iteration+1;
    startTime = clock();

    sigma[0] = {alpha*sigma0 + (1.-alpha)*sigma1};

    for (iteration; iteration <= maxiterations; iteration++)
        nlm.Run_GaussianNoise_Stack(path_rawdata, path_resdata, sigma, iteration, slices);

    endTime = clock();
    clockTicksTaken = endTime - startTime;
    timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
    cout << "\n\rTook " << timeInSeconds/((float) (slices*performed_iters)) << "s per slice and iteration" << endl;

    if ("append logfile"){
        time_t now = time(0);
        ofstream logfile;
        logfile.open(rootpath + "/logfile.txt", fstream::in | fstream::out | fstream::app);
        logfile << ctime(&now);
        logfile << "ran spring8_microtomodenoise:\n";
        logfile << "-------------------------------------------------------------------------\n";
        logfile << "    sigma: (" << sigma0 <<" / " << sigma1 << ")\n";
        logfile << "    alpha:  " << alpha << "\n";
        logfile << "    iterations: " << maxiterations << "\n";
        logfile << "    slices: " << nlm.nslices << "\n";
        logfile << "    patch space:  (" << nlm.patch_radius[0] << "," << nlm.patch_radius[1] << ","<< nlm.patch_radius[2] << ")\n";
        logfile << "    search space: (" << nlm.search_radius[0] << "," << nlm.search_radius[1] << "," << nlm.search_radius[2] << ")\n";
        logfile << "-------------------------------------------------------------------------\n\n";
        logfile.close();
    }

    /////////////////////////////////////////////////

    return 0;
}

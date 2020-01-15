#include <iostream>
#include <unistd.h>
#include <chrono>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>

#include "denoise_parameters.h"

#include "Geometry/hdcommunication.h"
#include "Geometry/auxiliary.h"

#include "Noise/noiselevel.h"
#include "Denoiser/iternlm_cpu.h"
#include "Color/color_denoising.h"

using namespace std;

/*********************************************************************************************************************************************************
 *
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *
 *********************************************************************************************************************************************************/

int main(int argc, char* argv[])
{
	string inpath = "";
	string outpath = "";

	bool verbose = false;
	/////////////////////////////////////////////////////////////////////

	protocol::DenoiseParameters params;
	string rootpath;

	if ("extract command line arguments)"){
		for (uint16_t i = 1; i < argc; i++)
		{
			//Data
			/////////////////////////////////////////////////////////////////////
			if ((string(argv[i]) == "-i") || (string(argv[i]) == "-input")){
				i++;
				inpath = string(argv[i]);
			}
			else if((string(argv[i]) == "-o") || (string(argv[i]) == "-output")){
				i++;
				outpath = string(argv[i]);
			}
			else if ((string(argv[i]) == "-r") || (string(argv[i]) == "-resume")){
				i++;
				if ((string(argv[i]) == "false") || (string(argv[i]) == "False"))
					params.io.resume = false;
				else if ((string(argv[i]) == "true") || (string(argv[i]) == "True"))
					params.io.resume = true;
			}
			else if (string(argv[i]) == "--resume") params.io.resume = true;
			else if ((string(argv[i]) == "-fs") || (string(argv[i]) == "-firstslice")){
				i++;
				params.io.firstslice = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-ls") || (string(argv[i]) == "-lastslice")){
				i++;
				params.io.lastslice = atoi(argv[i]); //lastslice is inclusive
			}
			else if ((string(argv[i]) == "--16bit")) params.io.save_type = "16bit";
			else if ((string(argv[i]) == "--cleanup")) params.io.cleanup = true;
			else if ((string(argv[i]) == "--color") || (string(argv[i]) == "--rgb")) params.io.rgb = true;
			//Denoising
			/////////////////////////////////////////////////////////////////////
			else if((string(argv[i]) == "-maxiter") || (string(argv[i]) == "-iter") || (string(argv[i]) == "-iterations")){
				i++;
				params.maxiterations = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-slices") || (string(argv[i]) == "-nslices"))
			{
				i++;
				params.nslices = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-a") || (string(argv[i]) == "-alpha"))
			{
				i++;
				params.alpha = atof(argv[i]);
			}
			else if ((string(argv[i]) == "-search")){
				i++;
				params.radius_searchspace[0] = atoi(argv[i]);
				params.radius_searchspace[1] = atoi(argv[i]);
				params.radius_searchspace[2] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-search0")){
				i++;
				params.radius_searchspace[0] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-search1")){
				i++;
				params.radius_searchspace[1] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-search2")){
				i++;
				params.radius_searchspace[2] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-patch")){
				i++;
				params.radius_patchspace[0] = atoi(argv[i]);
				params.radius_patchspace[1] = atoi(argv[i]);
				params.radius_patchspace[2] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-patch0")){
				i++;
				params.radius_patchspace[0] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-patch1")){
				i++;
				params.radius_patchspace[1] = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-patch2")){
				i++;
				params.radius_patchspace[2] = atoi(argv[i]);
			}
			else if (string(argv[i]) == "--independent") params.color.independent_channels = true;
			//Hardware
			/////////////////////////////////////////////////////////////////////
			else if ((string(argv[i]) == "-cpus") || (string(argv[i]) == "-n_cpus")){
				i++;
				params.cpu.max_threads = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "-gpus") || (string(argv[i]) == "-n_gpus") || (string(argv[i]) == "-ngpus")){
				i++;
				params.gpu.n_gpus = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-gpu0"){
				i++;
				params.gpu.deviceID = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-threads"){
				i++;
				params.gpu.threadsPerBlock = atoi(argv[i]);
			}
			else if ((string(argv[i]) == "--noblocks")){
				params.cpu.blockwise = false; //no blockwise processing in CPU mode
				params.gpu.blockwise_host = false; //no blockwise processing in GPU mode
			}
			else if ((string(argv[i]) == "--blocks")){
				params.cpu.blockwise = true; //blockwise processing in CPU mode
				params.gpu.blockwise_host = true; //blockwise processing in GPU mode
			}
			else if((string(argv[i]) == "--unroll"))
				params.cpu.unrolled_patchspace = true;
			else if((string(argv[i]) == "--verbose") || (string(argv[i]) == "--v")) verbose = true;
			//Noise Estimation
			/////////////////////////////////////////////////////////////////////
			else if((string(argv[i]) == "-noise") || (string(argv[i]) == "-noisemode")){
				i++;
				params.noiselevel.mode = string(argv[i]);
			}
			else if((string(argv[i]) == "-sigma") || (string(argv[i]) == "-s")){
				i++;
				params.noiselevel.sigma[0] = atof(argv[i]);
				params.noiselevel.sigma[1] = params.noiselevel.sigma[0];
			}
			else if((string(argv[i]) == "-sigma0") || (string(argv[i]) == "-s0")){
				i++;
				params.noiselevel.sigma[0] = atof(argv[i]);
			}
			else if((string(argv[i]) == "-sigma1") || (string(argv[i]) == "-s1")){
				i++;
				params.noiselevel.sigma[1] = atof(argv[i]);
			}
			else if ((string(argv[i]) == "-nsamples") || (string(argv[i]) == "-noise_samples")){
				i++;
				params.noiselevel.n_samples = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-noisepatch"){
				i++;
				params.noiselevel.patchsize = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-noiseshift"){
				i++;
				params.noiselevel.stds_from_mean = atof(argv[i]);
			}
			else if ((string(argv[i]) == "-mask") || (string(argv[i]) == "-mask_diameter")){
				i++;
				params.noiselevel.circular_mask_diameter = atoi(argv[i]);
			}
			else if (string(argv[i]) == "--nopoisson") params.color.stabilize_variance = false;
			else if (string(argv[i]) == "--continuous") params.noiselevel.continuous_estimate = true;
			else if (string(argv[i]) == "--noaverage") params.color.average_channelsigma = false;
			/////////////////////////////////////////////////////////////////////
			else if (inpath.length() == 0) inpath = string(argv[i]);
			else if (outpath.length() == 0) outpath = string(argv[i]);
			else if (string(argv[i]) == "-h" || string(argv[i]) == "-help" || string(argv[i]) == "--help")
			{
				cout << "Instructions for using iterNLM_v0.3 are provided on https://github.com/iternlm/microtomodenoise" << endl;
				return 0;
			}
		}
		rootpath = inpath.substr(0, inpath.rfind("/", inpath.length()-2)+1);
		if (outpath.length() == 0){
			rootpath = inpath.substr(0, inpath.rfind("/", inpath.length()-2)+1);
			outpath = rootpath + "/denoised/";}
		else
			rootpath = outpath.substr(0, outpath.rfind("/", outpath.length()-2)+1);
	}

	//set amount of cpu-threads (= amount of slices processed in parallel when using save_memory mode)
	if (params.cpu.max_threads > 0){
		params.cpu.max_threads = min(params.cpu.max_threads, omp_get_max_threads());
		omp_set_num_threads(params.cpu.max_threads);

	}
	else params.cpu.max_threads = omp_get_max_threads();

	//compiled for CPU:
	params.gpu.n_gpus = 0;

	auto time0 = chrono::high_resolution_clock::now();
	/////////////////////////////////////////////////////////////////////

	//read in raw data
	/////////////////////////////////////////////////////////////////////
	int shape[3];
	hdcom::HdCommunication hdcom;
	float *instack, *output;
	bool is_rgb;
	vector<string> filelist = hdcom.GetFilelist_And_ImageSequenceDimensions(inpath, shape, is_rgb);
	vector<string> filelist_prev;

	if (filelist[0] == "missing")
	{
		cout << "Error! Directory or file not found! Please provide a valid input with the -i argument "
				"or check https://github.com/iternlm/microtomodenoise for further instruction" << endl;
		return -1;
	}
	else if (filelist[0] == "no tif")
	{
		cout << "Error! No tif-file in directory! Please provide a valid input with the -i argument "
				"or check https://github.com/iternlm/microtomodenoise for further instruction" << endl;
		return -1;
	}

	if (params.io.rgb)
	{
		cout << "Switching to RGB-Denoiser!" << endl;
		color_denoise::run_colorimage_denoising(filelist, outpath, &params);
		return 0;
	}

	if (shape[2] == 1)
	{
		params.radius_patchspace[2] = 0;
		params.radius_searchspace[2] = 0;
	}

	cout << "--------------------------------------------------" << endl;
	cout << "iterNLM_v0.3 for " << inpath.substr((rootpath.rfind("/", rootpath.length()-2)+1), inpath.length()) << std::endl;
	cout << "--------------------------------------------------" << endl;
	cout << "alpha: " << params.alpha << endl;
	cout << "search_radius: " << params.radius_searchspace[0] << " " << params.radius_searchspace[1] << " " << params.radius_searchspace[2] << std::endl;
	cout << "n_slices: " << params.nslices << std::endl;
	cout << "patch_radius: " << params.radius_patchspace[0] << " " << params.radius_patchspace[1] << " " << params.radius_patchspace[2] << std::endl;
	cout << "noise estimate: " << params.noiselevel.mode << std::endl;
	cout << "image shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
	cout << "--------------------------------------------------" << endl;

	denoise::IterativeNLM_CPU iternlm;
	if (verbose){
		iternlm.print_estimatedmemory(shape, &params);
		cout << "--------------------------------------------------" << endl;
	}

	if (params.io.firstslice != -1 || params.io.lastslice != -1)
	{
		params.io.firstslice = std::max(0, params.io.firstslice);
		params.io.lastslice = std::min(shape[2]-1, params.io.lastslice);

		if (!params.cpu.blockwise  || (params.gpu.n_gpus > 0 && !params.gpu.blockwise_host))
			instack = hdcom.Get3DTifSequence_32bitPointer(filelist,shape,params.io.firstslice,params.io.lastslice);
		shape[2] = params.io.lastslice-params.io.firstslice+1;
	}
	else
	{
		params.io.firstslice = 0;
		params.io.lastslice = shape[2]-1;

		if (!params.cpu.blockwise || (params.gpu.n_gpus > 0 && !params.gpu.blockwise_host))
			instack = hdcom.GetTif_unknowndim_32bit(inpath, shape, true);
	}
	/////////////////////////////////////////////////////////////////////

	//estimate noise level
	/////////////////////////////////////////////////////////////////////
	float *sigma0, *sigma1;
	noise::NoiseLevel noise(params.noiselevel.n_samples, params.noiselevel.patchsize, shape);

	if ("get initial noise level"){

		bool estimate_available = false;
		if (params.io.resume)
			estimate_available = hdcom.ReadNoisefile(sigma0, shape, outpath + "/noise_vs_slice0.csv");

		if (!estimate_available)
		{
			cout << "estimating initial noise level...\r";
			cout.flush();

			sigma0 = noise.get_noiselevel(instack, filelist, &params);

			if(!params.io.cleanup || verbose)
				hdcom.SaveNoisefile(sigma0, shape[2], outpath, "noise_vs_slice0");
		}
		else cout << "Read in initial noise level from file!" << endl;

		cout << "initial noise level (slice " << shape[2]/2 << "): " << sigma0[shape[2]/2] << "          " << endl;
	}
	/////////////////////////////////////////////////////////////////////

	//select one of the protocols and run the denoising iterations
	/////////////////////////////////////////////////////////////////////
	if (1 == params.maxiterations && params.io.cleanup) params.io.active_outpath = outpath;
	else params.io.active_outpath = outpath+"/denoise-iteration01/";

	if (params.cpu.unrolled_patchspace && params.cpu.blockwise)
	{
		iternlm.Run_GaussianNoise_blocks_unrolled(1, filelist, filelist, sigma0, shape, &params);
		filelist_prev = hdcom.GetFilelist(params.io.active_outpath, shape);
	}
	else if (!params.cpu.unrolled_patchspace && params.cpu.blockwise)
	{
		iternlm.Run_GaussianNoise_blocks(1, filelist, filelist, sigma0, shape, &params);
		filelist_prev = hdcom.GetFilelist(params.io.active_outpath, shape);
	}
	else if (params.cpu.unrolled_patchspace)
	{
		output = iternlm.Run_GaussianNoise_unrolled(1, instack, output, sigma0, shape, &params);

		if(!params.io.cleanup || params.maxiterations == 1){
			if (params.io.save_type == "16bit") hdcom.SaveTifSequence_as16bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
			else hdcom.SaveTifSequence_32bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
		}
	}
	else{
		output = iternlm.Run_GaussianNoise(1, instack, output, sigma0, shape, &params);
		if(!params.io.cleanup || params.maxiterations == 1){
			if (params.io.save_type == "16bit") hdcom.SaveTifSequence_as16bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
			else hdcom.SaveTifSequence_32bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);

		}
	}
	/////////////////////////////////////////////////////////////////////

	//reestimate and weight the noise level
	/////////////////////////////////////////////////////////////////////
	if ("get texture level" && params.maxiterations > 1){

		bool estimate_available = false;
		if (params.io.resume)
			estimate_available = hdcom.ReadNoisefile(sigma1, shape, outpath + "/noise_vs_slice1.csv");

		if (!estimate_available)
		{
			cout << "reestimating noise level...\r";
			cout.flush();

			sigma1 = noise.get_noiselevel(output, filelist_prev, &params);

			if(!params.io.cleanup || verbose)
				hdcom.SaveNoisefile(sigma1, shape[2], outpath, "noise_vs_slice1");
		}
		else cout << "Read in initial noise level from file!" << endl;

		cout << "2nd noise level (slice " << shape[2]/2 << "): " << sigma1[shape[2]/2] << "          " << endl;

		//apply alpha weighting
		for (int i = 0; i < shape[2]; i++)
			sigma1[i] = params.alpha*sigma0[i] + (1.f-params.alpha)*sigma1[i];
	}
	/////////////////////////////////////////////////////////////////////

	//run remaining denoising iteration
	/////////////////////////////////////////////////////////////////////
	for (int iter = 2; iter <= params.maxiterations; iter++)
	{
		if (iter == params.maxiterations && params.io.cleanup) params.io.active_outpath = outpath;
		else params.io.active_outpath = outpath+"/denoise-iteration" + aux::zfill_int2string(iter, 2) + "/";

		if (params.cpu.unrolled_patchspace && params.cpu.blockwise)
		{
			iternlm.Run_GaussianNoise_blocks_unrolled(iter, filelist, filelist_prev, sigma1, shape, &params);
			filelist_prev = hdcom.GetFilelist(params.io.active_outpath, shape);
		}
		else if (!params.cpu.unrolled_patchspace && params.cpu.blockwise)
		{
			iternlm.Run_GaussianNoise_blocks(iter, filelist, filelist_prev, sigma1, shape, &params);
			filelist_prev = hdcom.GetFilelist(params.io.active_outpath, shape);
		}
		else if (params.cpu.unrolled_patchspace)
		{
			output = iternlm.Run_GaussianNoise_unrolled(iter, instack, output, sigma1, shape, &params);

			if(!params.io.cleanup || params.maxiterations == iter){
				if (params.io.save_type == "16bit") hdcom.SaveTifSequence_as16bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
				else hdcom.SaveTifSequence_32bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
			}
		}
		else
		{
			output = iternlm.Run_GaussianNoise(iter, instack, output, sigma1, shape, &params);

			if(!params.io.cleanup || params.maxiterations == iter){
				if (params.io.save_type == "16bit") hdcom.SaveTifSequence_as16bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
				else hdcom.SaveTifSequence_32bit(params.io.firstslice, output, shape, params.io.active_outpath, "denoised", false);
			}
		}
	}
	/////////////////////////////////////////////////////////////////////

	//remove obsolete directories
	/////////////////////////////////////////////////////////////////////
	if (params.io.cleanup)
	{
		for (int iter = 1; iter < params.maxiterations; iter++)
		{
			string obsolete_path = outpath+"/denoise-iteration" + aux::zfill_int2string(iter, 2) + "/";
			struct stat info;
			if (stat(obsolete_path.c_str(), &info) == 0){
				int sysreturn = system(("rm -r " +obsolete_path).c_str());
				if(sysreturn != 0);
			}
		}
	}
	/////////////////////////////////////////////////////////////////////

	if ("append logfile"){
		time_t now = time(0);
		ofstream logfile;
		logfile.open(rootpath + "/logfile.txt", fstream::in | fstream::out | fstream::app);
		logfile << ctime(&now);
		logfile << "ran iterNLM_v0.3 for " << inpath << "\n";
		logfile << "-------------------------------------------------------------------------\n";
		if(params.noiselevel.mode == "z-adaptive")
		{
			logfile << "    noise estimate: z-adaptive (moving window " << params.noiselevel.zwindow << ")\n";
			logfile << "    sigma0(slice "<<shape[2]/2<<"): " << sigma0[shape[2]/2] << "\n";
			if(params.maxiterations > 1) logfile << "    sigma1(slice "<<shape[2]/2<<"): " << sigma1[shape[2]/2] << "\n";
		}
		else
		{
			logfile << "    noise estimate: " << params.noiselevel.mode << "\n";
			logfile << "    sigma0: " << sigma0[shape[2]/2] << "\n";
			if(params.maxiterations > 1) logfile << "    sigma1: " << sigma1[shape[2]/2] << "\n";
		}
		logfile << "    alpha:  " << params.alpha << "\n";
		logfile << "    iterations: " << params.maxiterations << "\n";
		logfile << "    slices: " << params.nslices << "\n";
		logfile << "    patch space:  (" << params.radius_patchspace[0] << "," << params.radius_patchspace[1] << ","<< params.radius_patchspace[2] << ")\n";
		logfile << "    search space: (" << params.radius_searchspace[0] << "," << params.radius_searchspace[1] << "," << params.radius_searchspace[2] << ")\n";
		logfile << "-------------------------------------------------------------------------\n\n";
		logfile.close();
	}
	cout << "--------------------------------------------------" << endl;
	auto time_final = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed_total = time_final-time0;
	std::cout << "execution took " << elapsed_total.count() << " s" << std::endl;
	cout << "--------------------------------------------------" << endl;

	return 0;
}

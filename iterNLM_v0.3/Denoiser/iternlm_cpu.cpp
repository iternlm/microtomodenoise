#include "iternlm_cpu.h"

#include "../Geometry/hdcommunication.h"
#include "../Geometry/auxiliary.h"

namespace denoise
{
	float* IterativeNLM_CPU::Run_GaussianNoise(int iter, float* &image_raw, float* &previous_result, float* sigmalist, int shape[3], protocol::DenoiseParameters *params)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		int shape_padded[3]; //remembers the shape of the padded data

		//apply padding
		//////////////////////////////////////////////////////////////////////////////
		int padding[6] = {params->radius_searchspace[0]+params->radius_patchspace[0],
				          params->radius_searchspace[1]+params->radius_patchspace[1],
				          std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2],
				          params->radius_searchspace[0]+params->radius_patchspace[0],
						  params->radius_searchspace[1]+params->radius_patchspace[1],
						  std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2]};

		if (iter == 1){
			float *tmp = pad_reflective(image_raw, padding, shape, shape_padded);
			std::swap(tmp, image_raw);
			free(tmp);

			previous_result = image_raw;
		}
		else
		{
			float *tmp = pad_reflective(previous_result, padding, shape, shape_padded);

			std::swap(tmp, previous_result);
			free(tmp);
		}
		//////////////////////////////////////////////////////////////////////////////

		if (iter == 1)
		{
			search_positions = setup_searchspace(shape_padded, params, nsize_search);
			patch_positions = setup_patchspace(shape_padded, params, nsize_patch);
			distweight = setup_distweight(shape_padded, params);
		}

		float* next_result = (float*) malloc(nstack*sizeof(*next_result));

		//Check if iteration is already complete
		/////////////////////////////////////////////////////////////////////////////
		if (params->io.resume)
		{
			struct stat buffer;
			bool finished_iteration = true;

			for (int i = params->io.firstslice; i <= params->io.lastslice; i++)
			{
				//file exists?
				std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i,4)+".tif";
				if (stat (filename.c_str(), &buffer) != 0){finished_iteration = false; break;}

				//file is complete?
				std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
				const auto filesize = testFile.tellg();
				if (filesize > expected_filesize) expected_filesize = filesize;
				else if (filesize != expected_filesize){finished_iteration = false; break;}
			}
			if (finished_iteration)
			{
				hdcom::HdCommunication hdcom;
				int tmpshape[3];

				std::vector<std::string> filelist = hdcom.GetFilelist(params->io.active_outpath, tmpshape);
				std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";
				int this_fs = 0;

				for (int i = 0; i < (int) filelist.size(); i++)
					if (filelist[i].substr(filelist[i].length()-8,8) == fs_sub){this_fs = i; break;}
				int this_ls = this_fs + (params->io.lastslice-params->io.firstslice);

				next_result = hdcom.Get3DTifSequence_32bitPointer(filelist, shape, this_fs, this_ls);
				std::cout << "iteration " << iter << " read in from disk" << std::endl;
				return next_result;
			}
		}
		/////////////////////////////////////////////////////////////////////////////

		int evalcounter = 0;
		auto time0 = std::chrono::high_resolution_clock::now();

		#pragma omp parallel for
		for (int i = 0; i < shape[2]; i++)
		{
			float sigma = sigmalist[i];
			float multiplier = -1.f/(sigma*sigma*params->beta); //changes depending on way of implementation. Beta is available if control needed

			if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
				filterslice_p111(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
				filterslice_p112(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
				filterslice_p113(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
				filterslice_p221(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
				filterslice_p222(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 1)
				filterslice_p331(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 2)
				filterslice_p332(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 3 && params->radius_patchspace[1] == 3 && params->radius_patchspace[2] == 3)
				filterslice_p333(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else
				filterslice(i, multiplier, image_raw, previous_result, next_result, shape, params);

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			int tid = omp_get_thread_num();
			if (tid == 0)
			{
				evalcounter++;
				auto time_final = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed_total = time_final-time0;
				std::cout << "iteration " << iter << ": " << std::min(shape[2], evalcounter*params->cpu.max_threads) << "/" << shape[2] << ", "
						<< round(elapsed_total.count()/evalcounter*10.)/(10.f*params->cpu.max_threads)*(shape[2]-std::min(shape[2], evalcounter*params->cpu.max_threads))
						<< " s remaining          \r";
				std::cout.flush();
			}
			////////////////////////////////////////////////////////////////////////////////////////////
		}

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << round(elapsed_total.count()*10.)/10. << " s                          " << std::endl;

		if(iter != 1) free(previous_result);
		//////////////////////////////////////////////////////////////////////////////

		return next_result;
	}
	float* IterativeNLM_CPU::Run_GaussianNoise_unrolled(int iter, float* &image_raw, float* &previous_result, float* sigmalist, int shape[3], protocol::DenoiseParameters *params)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		int shape_padded[3]; //remembers the shape of the padded data

		if (iter == 1)
			patch_positions = setup_patchspace(shape, params, nsize_patch); //patch positions in unpadded data

		//apply padding and unroll
		//////////////////////////////////////////////////////////////////////////////
		int padding[6] = {params->radius_searchspace[0],params->radius_searchspace[1],std::min(params->nslices/2, params->radius_searchspace[2]),
						  params->radius_searchspace[0],params->radius_searchspace[1],std::min(params->nslices/2, params->radius_searchspace[2])};

		if (iter == 1)
		{
			previous_result = pad_reflective_unrollpatchspace(image_raw, padding, shape, shape_padded, patch_positions, nsize_patch);
			float* tmp = pad_reflective(image_raw, padding, shape, shape_padded);

			std::swap(tmp, image_raw);
			free(tmp);
		}
		else
		{
			float* tmp = pad_reflective_unrollpatchspace(previous_result, padding, shape, shape_padded, patch_positions, nsize_patch);

			std::swap(previous_result, tmp);
			free(tmp);
		}
		//////////////////////////////////////////////////////////////////////////////

		if (iter == 1)
		{
			search_positions = setup_searchspace(shape_padded, params, nsize_search);
			distweight = setup_distweight(shape_padded, params);
		}

		float* next_result = (float*) malloc(nstack*sizeof(*next_result));

		//Check if iteration is already complete
		/////////////////////////////////////////////////////////////////////////////
		if (params->io.resume)
		{
			struct stat buffer;
			bool finished_iteration = true;

			for (int i = params->io.firstslice; i <= params->io.lastslice; i++)
			{
				//file exists?
				std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i,4)+".tif";
				if (stat (filename.c_str(), &buffer) != 0){finished_iteration = false; break;}

				//file is complete?
				std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
				const auto filesize = testFile.tellg();
				if (filesize > expected_filesize) expected_filesize = filesize;
				else if (filesize != expected_filesize){finished_iteration = false; break;}
			}
			if (finished_iteration)
			{
				hdcom::HdCommunication hdcom;
				int tmpshape[3];

				std::vector<std::string> filelist = hdcom.GetFilelist(params->io.active_outpath, tmpshape);
				std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";
				int this_fs = 0;

				for (int i = 0; i < (int) filelist.size(); i++)
					if (filelist[i].substr(filelist[i].length()-8,8) == fs_sub){this_fs = i; break;}
				int this_ls = this_fs + (params->io.lastslice-params->io.firstslice);

				next_result = hdcom.Get3DTifSequence_32bitPointer(filelist, shape, this_fs, this_ls);
				std::cout << "iteration " << iter << " read in from disk" << std::endl;
				return next_result;
			}
		}
		/////////////////////////////////////////////////////////////////////////////

		int evalcounter = 0;
		auto time0 = std::chrono::high_resolution_clock::now();

		#pragma omp parallel for
		for (int i = 0; i < shape[2]; i++)
		{
			float sigma = sigmalist[i];
			float multiplier = -1.f/(sigma*sigma*params->beta); //changes depending on way of implementation. Beta is available if control needed

			if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
				filterslice_p111_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
				filterslice_p112_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
				filterslice_p113_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
				filterslice_p221_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
				filterslice_p222_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
				filterslice_p331_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
				filterslice_p332_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 3)
				filterslice_p333_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);
			else
				filterslice_unrolled(i, multiplier, image_raw, previous_result, next_result, shape, params);

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			int tid = omp_get_thread_num();
			if (tid == 0)
			{
				evalcounter++;
				auto time_final = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed_total = time_final-time0;
				std::cout << "iteration " << iter << ": " << std::min(shape[2], evalcounter*params->cpu.max_threads) << "/" << shape[2] << ", "
						<< round(elapsed_total.count()/evalcounter*10.)/(10.f*params->cpu.max_threads)*(shape[2]-std::min(shape[2], evalcounter*params->cpu.max_threads))
						<< " s remaining          \r";
				std::cout.flush();
			}
			////////////////////////////////////////////////////////////////////////////////////////////
		}

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << round(elapsed_total.count()*10.)/10. << " s                          " << std::endl;
		//////////////////////////////////////////////////////////////////////////////

		free(previous_result);
		return next_result;
	}
	void IterativeNLM_CPU::Run_GaussianNoise_3channels(int iter, std::vector<float*> &raw_channels, std::vector<float*> &previous_results, std::vector<float> &channel_sigma,
			int shape[3], protocol::DenoiseParameters *params)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		int shape_padded[3]; //remembers the shape of the padded data

		//apply padding
		//////////////////////////////////////////////////////////////////////////////
		int padding[6] = {params->radius_searchspace[0]+params->radius_patchspace[0],
						  params->radius_searchspace[1]+params->radius_patchspace[1],
						  std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2],
						  params->radius_searchspace[0]+params->radius_patchspace[0],
						  params->radius_searchspace[1]+params->radius_patchspace[1],
						  std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2]};

		if (iter == 1){
			float *tmp0 = pad_reflective(raw_channels[0], padding, shape, shape_padded);
			float *tmp1 = pad_reflective(raw_channels[1], padding, shape, shape_padded);
			float *tmp2 = pad_reflective(raw_channels[2], padding, shape, shape_padded);
			std::swap(tmp0, raw_channels[0]);
			std::swap(tmp1, raw_channels[1]);
			std::swap(tmp2, raw_channels[2]);
			free(tmp0); free(tmp1); free(tmp2);

			previous_results.push_back(raw_channels[0]);
			previous_results.push_back(raw_channels[1]);
			previous_results.push_back(raw_channels[2]);
		}
		else
		{
			float *tmp0 = pad_reflective(previous_results[0], padding, shape, shape_padded);
			float *tmp1 = pad_reflective(previous_results[1], padding, shape, shape_padded);
			float *tmp2 = pad_reflective(previous_results[2], padding, shape, shape_padded);

			std::swap(tmp0, previous_results[0]);
			std::swap(tmp1, previous_results[1]);
			std::swap(tmp2, previous_results[2]);
			free(tmp0); free(tmp1); free(tmp2);
		}
		//////////////////////////////////////////////////////////////////////////////

		if (iter == 1)
		{
			search_positions = setup_searchspace(shape_padded, params, nsize_search);
			patch_positions = setup_patchspace(shape_padded, params, nsize_patch);
			distweight = setup_distweight(shape_padded, params);
		}

		//generate kernel code
		if (0 == 1){
			std::ofstream outfile;
			outfile.open("/home/stefan/Documents/kernelcode.txt");

			outfile << "patchsize: " << nsize_patch << "\n";

			for (int i = 0; i < nsize_patch; i++)
			{
				if (i == 0) continue;
				else outfile << "idx_type ppos"<<i<< "= patch_positions["<<i<<"];";
				if (i != 0 && (i%4) == 0) outfile << "\n";
			}
			outfile << "\n-----------------------------\n";
			for (int i = 0; i < nsize_patch; i++)
			{
				if (i == 0){
					outfile << "float val_orig0_0 = images_prefiltered[0][idx0        ];";
					outfile << "float val_orig1_0 = images_prefiltered[1][idx0        ];";
					outfile << "float val_orig2_0 = images_prefiltered[2][idx0        ];";
				}
				else
				{
					outfile << "float val_orig0_"<<i<<"=images_prefiltered[0][idx0+ppos"<<i<<"];";
					outfile << "float val_orig1_"<<i<<"=images_prefiltered[1][idx0+ppos"<<i<<"];";
					outfile << "float val_orig2_"<<i<<"=images_prefiltered[2][idx0+ppos"<<i<<"];";
				}
				if (i != 0 && (i%4) == 0) outfile << "\n";
			}
			outfile << "\n-----------------------------\n";
			for (int i = 0; i < nsize_patch; i++)
			{
				if (i== 0){
					outfile << "float tmp = images_prefiltered[0][idx1        ]-val_orig0_0; distance0 += (tmp*tmp)*" << distweight[i] << "f;\n";
					outfile << "      tmp = images_prefiltered[1][idx1        ]-val_orig1_0; distance1 += (tmp*tmp)*" << distweight[i] << "f;\n";
					outfile << "      tmp = images_prefiltered[2][idx1        ]-val_orig2_0; distance2 += (tmp*tmp)*" << distweight[i] << "f;\n";
				}
				else
				{
					outfile<< "tmp = images_prefiltered[0][idx1 + ppos" << i << "]-val_orig0_"<<i<<"; distance0 += (tmp*tmp)*" << distweight[i] << "f;\n";
					outfile<< "tmp = images_prefiltered[1][idx1 + ppos" << i << "]-val_orig1_"<<i<<"; distance1 += (tmp*tmp)*" << distweight[i] << "f;\n";
					outfile<< "tmp = images_prefiltered[2][idx1 + ppos" << i << "]-val_orig2_"<<i<<"; distance2 += (tmp*tmp)*" << distweight[i] << "f;\n";
				}
			}
			outfile << "\n-----------------------------\n";
			outfile.close();
		}

		float* next_result0 = (float*) malloc(nstack*sizeof(*next_result0));
		float* next_result1 = (float*) malloc(nstack*sizeof(*next_result1));
		float* next_result2 = (float*) malloc(nstack*sizeof(*next_result2));
		std::vector<float*> next_results = {next_result0, next_result1, next_result2};

		auto time0 = std::chrono::high_resolution_clock::now();

		float multiplier[3] = {-1.f/(channel_sigma[0]*channel_sigma[0]*params->beta),
				               -1.f/(channel_sigma[1]*channel_sigma[1]*params->beta),
				               -1.f/(channel_sigma[2]*channel_sigma[2]*params->beta)};

		filterslice_3channels(0, multiplier, raw_channels, previous_results, next_results, shape, params);

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << round(elapsed_total.count()*1000.)/1000. << " s                          " << std::endl;

		if(iter != 1)
		{
			free(previous_results[0]);
			free(previous_results[1]);
			free(previous_results[2]);
		}
		previous_results.clear();

		previous_results.push_back(next_result0);
		previous_results.push_back(next_result1);
		previous_results.push_back(next_result2);
		//////////////////////////////////////////////////////////////////////////////*/

		return;
	}

	void IterativeNLM_CPU::Run_GaussianNoise_blocks(int iter, std::vector<std::string> &filelist_raw, std::vector<std::string> &filelist_prev, float* sigmalist,
			int shape[3], protocol::DenoiseParameters *params)
	{
		//check if only substack processing and get offset in filelists
		/////////////////////////////////////////////////////////////////
		int firstslice_offset[2] = {0, 0};

		if (params->io.firstslice != 0)
		{
			firstslice_offset[0] = params->io.firstslice;

			std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";

			for (int i = 0; i < (int) filelist_prev.size(); i++)
				if (filelist_prev[i].substr(filelist_prev[i].length()-8,8) == fs_sub){firstslice_offset[1] = i; break;}
		}
		/////////////////////////////////////////////////////////////////

		//set the block dimensions
		/////////////////////////////////////////////////////////////////
		int blocklength = params->cpu.max_threads;

		int n_blocks = shape[2]/blocklength;
		if ((shape[2]%blocklength) != 0) n_blocks++;

		int padding[6] = {params->radius_searchspace[0]+params->radius_patchspace[0],
						  params->radius_searchspace[1]+params->radius_patchspace[1],
						  std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2],
						  params->radius_searchspace[0]+params->radius_patchspace[0],
						  params->radius_searchspace[1]+params->radius_patchspace[1],
						  std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2]};
		int zpadding[2] = {padding[2], padding[5]};

		int blockshape[3] = {shape[0], shape[1], blocklength};
		long long int blockslice = blockshape[0]*blockshape[1];
		long long int blockstack = blockshape[2]*blockslice;

		int blockshape_padded[3] = {blockshape[0]+padding[0]+padding[3], blockshape[1]+padding[1]+padding[4], blockshape[2]+padding[2]+padding[5]};
		/////////////////////////////////////////////////////////////////

		//precalculate immutable values
		/////////////////////////////////////////////////////////////////
		if (iter == 1)
		{
			patch_positions = setup_patchspace(blockshape_padded, params, nsize_patch); //patch positions padded data
			search_positions = setup_searchspace(blockshape_padded, params, nsize_search);
			distweight = setup_distweight(blockshape_padded, params);
		}
		/////////////////////////////////////////////////////////////////

		float* output = (float*) malloc(blockstack*sizeof(*output));

		hdcom::HdCommunication hdcom;
		auto time0 = std::chrono::high_resolution_clock::now();
		int skipped_blocks = 0;

		for (int n = 0; n < n_blocks; n++)
		{
			int firstslice = n*blocklength-zpadding[0];
			int lastslice = std::min((n+1)*blocklength-1, shape[2]-1)+zpadding[1];

			//Check if block already fully processed and resume
			/////////////////////////////////////////////////////////////////////////////
			if (params->io.resume)
			{
				bool finished_block = true;
				struct stat buffer;

				for (int i = firstslice+zpadding[0]; i <= lastslice-zpadding[1]; i++)
				{
					//file exists?
					std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i+params->io.firstslice,4)+".tif";
					if (stat (filename.c_str(), &buffer) != 0){finished_block = false; break;}

					//file is complete?
					std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
					const auto filesize = testFile.tellg();
					if (filesize > expected_filesize) expected_filesize = filesize;
					else if (filesize != expected_filesize){finished_block = false; break;}
				}
				if (finished_block){skipped_blocks++; continue;}
			}
			/////////////////////////////////////////////////////////////////////////////

			//check if we need to pad in z-direction
			/////////////////////////////////////////////////////////////////////////////
			if (firstslice < 0) {padding[2] = -firstslice; firstslice = 0;}
			else padding[2] = 0;
			if (lastslice >= shape[2]) {padding[5] = lastslice-(shape[2]-1); lastslice = shape[2]-1;}
			else padding[5] = 0;
			/////////////////////////////////////////////////////////////////////////////

			//apply padding and unroll
			/////////////////////////////////////////////////////////////////////////////
			float *imageblock_raw, *imageblock_previous;

			if (iter == 1)
			{
				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
				imageblock_previous = pad_reflective(imageblock_raw, padding, blockshape, blockshape_padded);
			}
			else
			{
				float* tmp = hdcom.Get3DTifSequence_32bitPointer(filelist_prev, blockshape, firstslice+firstslice_offset[1], lastslice+firstslice_offset[1]);
				imageblock_previous = pad_reflective(tmp, padding, blockshape, blockshape_padded);

				free(tmp);

				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
			}
			float* tmp = pad_reflective(imageblock_raw, padding, blockshape, blockshape_padded);

			std::swap(tmp, imageblock_raw);
			free(tmp);
			/////////////////////////////////////////////////////////////////////////////

			#pragma omp parallel for
			for (int i = 0; i < blocklength; i++)
			{
				int z = n*blocklength + i;

				if (z < shape[2])
				{
					float sigma = sigmalist[z];
					float multiplier = -1.f/(sigma*sigma*params->beta); //changes depending on way of implementation. Beta is available if control needed

					if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
						filterslice_p111(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
						filterslice_p112(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
						filterslice_p113(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
						filterslice_p221(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
						filterslice_p222(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
						filterslice_p331(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
						filterslice_p332(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 3)
						filterslice_p333(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else
						filterslice(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
				}
			}

			blockshape[2] = std::min((n+1)*blocklength, shape[2])-(n*blocklength);
			if(params->io.save_type == "16bit") hdcom.SaveTifSequence_as16bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);
			else hdcom.SaveTifSequence_32bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);

			free(imageblock_raw);
			free(imageblock_previous);

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			auto time_final = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_total = time_final-time0;
			std::cout << "iteration " << iter << ": " << std::min(shape[2], (n+1)*params->cpu.max_threads) << "/" << shape[2] << ", "
					<< round(elapsed_total.count()/(n+1-skipped_blocks)*10.)/(10.f*params->cpu.max_threads)*
					  (shape[2]-std::min(shape[2]-skipped_blocks*blocklength, (n+1)*params->cpu.max_threads-skipped_blocks*blocklength))
					<< " s remaining          \r";
			std::cout.flush();
			////////////////////////////////////////////////////////////////////////////////////////////
		}

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << round(elapsed_total.count()*10.)/10. << " s                          " << std::endl;

		//////////////////////////////////////////////////////////////////////////////

		free(output);
		return;
	}
	void IterativeNLM_CPU::Run_GaussianNoise_blocks_unrolled(int iter, std::vector<std::string> &filelist_raw, std::vector<std::string> &filelist_prev, float* sigmalist,
			int shape[3], protocol::DenoiseParameters *params)
	{
		//check if only substack processing and get offset in filelists
		/////////////////////////////////////////////////////////////////
		int firstslice_offset[2] = {0, 0};

		if (params->io.firstslice != 0)
		{
			firstslice_offset[0] = params->io.firstslice;

			std::string fs_sub = aux::zfill_int2string(params->io.firstslice,4)+".tif";

			for (int i = 0; i < (int) filelist_prev.size(); i++)
				if (filelist_prev[i].substr(filelist_prev[i].length()-8,8) == fs_sub){firstslice_offset[1] = i; break;}
		}
		/////////////////////////////////////////////////////////////////

		//set the block dimensions
		/////////////////////////////////////////////////////////////////
		int blocklength = params->cpu.max_threads;

		int n_blocks = shape[2]/blocklength;
		if ((shape[2]%blocklength) != 0) n_blocks++;

		int padding[6] = {params->radius_searchspace[0],params->radius_searchspace[1],std::min(params->nslices/2, params->radius_searchspace[2]),
						  params->radius_searchspace[0],params->radius_searchspace[1],std::min(params->nslices/2, params->radius_searchspace[2])};
		int zpadding[2] = {padding[2], padding[5]};

		int blockshape[3] = {shape[0], shape[1], blocklength};
		long long int blockslice = blockshape[0]*blockshape[1];
		long long int blockstack = blockshape[2]*blockslice;

		int blockshape_padded[3] = {blockshape[0]+padding[0]+padding[3], blockshape[1]+padding[1]+padding[4], blockshape[2]+padding[2]+padding[5]};
		/////////////////////////////////////////////////////////////////

		//precalculate constant values
		/////////////////////////////////////////////////////////////////
		if (iter == 1)
		{
			patch_positions = setup_patchspace(blockshape, params, nsize_patch); //patch positions in unpadded data
			search_positions = setup_searchspace(blockshape_padded, params, nsize_search);
			distweight = setup_distweight(blockshape_padded, params);
		}
		/////////////////////////////////////////////////////////////////

		float* output = (float*) malloc(blockstack*sizeof(*output));

		hdcom::HdCommunication hdcom;
		auto time0 = std::chrono::high_resolution_clock::now();
		int skipped_blocks = 0;

		//loop over blocks
		/////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////
		for (int n = 0; n < n_blocks; n++)
		{
			int firstslice = n*blocklength-zpadding[0];
			int lastslice = std::min((n+1)*blocklength-1, shape[2]-1)+zpadding[1];

			//Check if block already fully processed and resume
			/////////////////////////////////////////////////////////////////////////////
			if (params->io.resume)
			{
				bool finished_block = true;
				struct stat buffer;

				for (int i = firstslice+zpadding[0]; i <= lastslice-zpadding[1]; i++)
				{
					//file exists?
					std::string filename = params->io.active_outpath + "/denoised" + aux::zfill_int2string(i+params->io.firstslice,4)+".tif";
					if (stat (filename.c_str(), &buffer) != 0){finished_block = false; break;}

					//file is complete?
					std::ifstream testFile(filename.c_str(), std::ios::binary | std::ios::ate);
					const auto filesize = testFile.tellg();
					if (filesize > expected_filesize) expected_filesize = filesize;
					else if (filesize != expected_filesize){finished_block = false; break;}
				}
				if (finished_block){skipped_blocks++; continue;}
			}
			/////////////////////////////////////////////////////////////////////////////

			//check if we need to pad in z-direction
			if (firstslice < 0) {padding[2] = -firstslice; firstslice = 0;}
			else padding[2] = 0;
			if (lastslice >= shape[2]) {padding[5] = lastslice-(shape[2]-1); lastslice = shape[2]-1;}
			else padding[5] = 0;

			//apply padding and unroll
			float *imageblock_raw, *imageblock_previous;

			//read in and pad substack
			/////////////////////////////////////////////////////////////////////////////
			if (iter == 1)
			{
				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
				imageblock_previous = pad_reflective_unrollpatchspace(imageblock_raw, padding, blockshape, blockshape_padded, patch_positions, nsize_patch);
			}
			else
			{
				float* tmp = hdcom.Get3DTifSequence_32bitPointer(filelist_prev, blockshape, firstslice+firstslice_offset[1], lastslice+firstslice_offset[1]);
				imageblock_previous = pad_reflective_unrollpatchspace(tmp, padding, blockshape, blockshape_padded, patch_positions, nsize_patch);
				free(tmp);

				imageblock_raw = hdcom.Get3DTifSequence_32bitPointer(filelist_raw, blockshape, firstslice+firstslice_offset[0], lastslice+firstslice_offset[0]);
			}
			float* tmp = pad_reflective(imageblock_raw, padding, blockshape, blockshape_padded);
			std::swap(tmp, imageblock_raw);
			free(tmp);
			/////////////////////////////////////////////////////////////////////////////

			//denoising
			/////////////////////////////////////////////////////////////////////////////
			#pragma omp parallel for
			for (int i = 0; i < blocklength; i++)
			{
				int z = n*blocklength + i;

				if (z < shape[2])
				{
					float sigma = sigmalist[z];
					float multiplier = -1.f/(sigma*sigma*params->beta); //changes depending on way of implementation. Beta is available if control needed

					if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 1)
						filterslice_p111_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 2)
						filterslice_p112_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 1 && params->radius_patchspace[1] == 1 && params->radius_patchspace[2] == 3)
						filterslice_p113_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
						filterslice_p221_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
						filterslice_p222_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 1)
						filterslice_p331_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 2)
						filterslice_p332_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else if (params->radius_patchspace[0] == 2 && params->radius_patchspace[1] == 2 && params->radius_patchspace[2] == 3)
						filterslice_p333_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
					else
						filterslice_unrolled(i, multiplier, imageblock_raw, imageblock_previous, output, shape, params);
				}
			}
			/////////////////////////////////////////////////////////////////////////////

			blockshape[2] = std::min((n+1)*blocklength, shape[2])-(n*blocklength);

			if(params->io.save_type == "16bit") hdcom.SaveTifSequence_as16bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);
			else hdcom.SaveTifSequence_32bit((n*blocklength)+params->io.firstslice, output, blockshape, params->io.active_outpath, "denoised", false);

			free(imageblock_raw);
			free(imageblock_previous);

			//console output
			////////////////////////////////////////////////////////////////////////////////////////////
			auto time_final = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_total = time_final-time0;
			std::cout << "iteration " << iter << ": " << std::min(shape[2], (n+1)*params->cpu.max_threads) << "/" << shape[2] << ", "
					<< round(elapsed_total.count()/(n+1-skipped_blocks)*10.)/(10.f*params->cpu.max_threads)*
					  (shape[2]-std::min(shape[2]-skipped_blocks*blocklength, (n+1)*params->cpu.max_threads-skipped_blocks*blocklength))
					<< " s remaining          \r";
			std::cout.flush();
			////////////////////////////////////////////////////////////////////////////////////////////
		}
		/////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////

		auto time_final = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_total = time_final-time0;
		std::cout << "iteration " << iter << " took " << round(elapsed_total.count()*10.)/10. << " s                          " << std::endl;
		//////////////////////////////////////////////////////////////////////////////

		free(output);
		return;
	}

	/*******************************************************************************************************************************************************************/
	/*******************************************************************************************************************************************************************/

	void IterativeNLM_CPU::filterslice(int z0, float multiplier, float* image_raw, float *image_prefiltered, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		long long int nslice = nx*ny;
		long long int offset = (z0+zpad)*nslice;
		long long int nslice_unpadded = shape[0]*shape[1];
		long long int offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				long long int idx0 = offset + y0*nx + x0;
				float noisy_value_origin = image_raw[idx0];

				//get patchvalues at origin
				/////////////////////////////////////////////////////////////////////////
				float* values_origin = (float*) malloc(nsize_patch*sizeof(*values_origin));

				for (int p = 0; p < nsize_patch; p++)
					values_origin[p] = image_prefiltered[idx0 + patch_positions[p]];
				/////////////////////////////////////////////////////////////////////////

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					long long int idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					for (int p = 0; p < nsize_patch; p++)
					{
						float tmp = image_prefiltered[idx1 + patch_positions[p]]-values_origin[p];
						distance += (tmp*tmp)*distweight[p];
					}
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

				free(values_origin);

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_unrolled(int z0, float multiplier, float* image_raw, float *image_prefiltered, float *result, int shape[3], protocol::DenoiseParameters *params)
	{
		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0];
		int ypad = params->radius_searchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2]);

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		long long int nslice = nx*ny;
		long long int offset = (z0+zpad)*nslice;
		long long int nslice_unpadded = shape[0]*shape[1];
		long long int offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				long long int idx0 = offset + y0*nx + x0;
				long long int idx0b = idx0*nsize_patch;
				float noisy_value_origin = image_raw[idx0];

				//get patchvalues at origin
				/////////////////////////////////////////////////////////////////////////
				float* values_origin = (float*) malloc(nsize_patch*sizeof(*values_origin));

				for (int p = 0; p < nsize_patch; p++)
					values_origin[p] = image_prefiltered[idx0b + p];
				/////////////////////////////////////////////////////////////////////////

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue = 0.0f;
				float filterweight = 0.0f;
				float maxweight = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					long long int idx1 = idx0 + search_positions[s];
					long long int idx1b = idx1*nsize_patch;
					float noisy_value_searchpos = image_raw[idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance = 0.0f;

					for (int p = 0; p < nsize_patch; p++)
					{
						float tmp = image_prefiltered[idx1b + p]-values_origin[p];
						distance += (tmp*tmp)*distweight[p];
					}
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

				free(values_origin);

				//continue image space
			}
		}

		return;
	}
	void IterativeNLM_CPU::filterslice_3channels(int z0, float multiplier[3], std::vector<float*> &raw_channels, std::vector<float*> &images_prefiltered, std::vector<float*> &results,
			int shape[3], protocol::DenoiseParameters *params)
	{
		//Estimating similarity from all three channels

		//Image space
		//////////////////////////////////////////////////////////////////////////////
		int xpad = params->radius_searchspace[0]+params->radius_patchspace[0];
		int ypad = params->radius_searchspace[1]+params->radius_patchspace[1];
		int zpad = std::min(params->nslices/2, params->radius_searchspace[2])+params->radius_patchspace[2];

		int nx = shape[0]+2*xpad; int ny = shape[1]+2*ypad;
		long long int nslice = nx*ny;
		long long int offset = (z0+zpad)*nslice;
		long long int nslice_unpadded = shape[0]*shape[1];
		long long int offset_unpadded = z0*nslice_unpadded;
		//////////////////////////////////////////////////////////////////////////////

		//loop over slice
		#pragma omp parallel for
		for (int y0 = ypad; y0 < ny-ypad; y0++)
		{
			for (int x0 = xpad; x0 < nx-xpad; x0++)
			{
				long long int idx0 = offset + y0*nx + x0;
				float noisy_value_origin0 = raw_channels[0][idx0];
				float noisy_value_origin1 = raw_channels[1][idx0];
				float noisy_value_origin2 = raw_channels[2][idx0];

				//get patchvalues at origin
				/////////////////////////////////////////////////////////////////////////
				float* values_origin = (float*) malloc(3*nsize_patch*sizeof(*values_origin));

				for (int p = 0; p < nsize_patch; p++)
				{
					idx_type ppos = patch_positions[p];
					values_origin[3*p  ] = images_prefiltered[0][idx0 + ppos];
					values_origin[3*p+1] = images_prefiltered[1][idx0 + ppos];
					values_origin[3*p+2] = images_prefiltered[2][idx0 + ppos];
				}
				/////////////////////////////////////////////////////////////////////////

				//loop over search space
				/////////////////////////////////////////////////////////////////////////
				float filtervalue0 = 0.0f; float filtervalue1 = 0.0f; float filtervalue2 = 0.0f;
				float filterweight0 = 0.0f; float filterweight1 = 0.0f; float filterweight2 = 0.0f;
				float maxweight0 = 0.0f; float maxweight1 = 0.0f; float maxweight2 = 0.0f;

				for (int s = 0; s < nsize_search; s++)
				{
					long long int idx1 = idx0 + search_positions[s];
					float noisy_value_searchpos0 = raw_channels[0][idx1];
					float noisy_value_searchpos1 = raw_channels[1][idx1];
					float noisy_value_searchpos2 = raw_channels[2][idx1];

					//get patchvalues at search position
					/////////////////////////////////////////////////////////////////////////
					float distance0 = 0.0f; float distance1 = 0.0f; float distance2 = 0.0f;

					for (int p = 0; p < nsize_patch; p++)
					{
						idx_type ppos = patch_positions[p];
						float dw = distweight[p];

						float tmp = images_prefiltered[0][idx1 + ppos]-values_origin[3*p  ]; distance0 += (tmp*tmp)*dw;
						      tmp = images_prefiltered[1][idx1 + ppos]-values_origin[3*p+1]; distance1 += (tmp*tmp)*dw;
						      tmp = images_prefiltered[2][idx1 + ppos]-values_origin[3*p+2]; distance2 += (tmp*tmp)*dw;
					}
					/////////////////////////////////////////////////////////////////////////

					//weight the patch
					/////////////////////////////////////////////////////////////////////////
					distance0 = distance0*multiplier[0];
					distance1 = distance1*multiplier[1];
					distance2 = distance2*multiplier[2];

					if (!params->color.independent_channels)
					{
						float distsum = (distance0+distance1+distance2)/3.f;
						float this_weight = (distsum > expapproximation_cutoff) ? expapproximation(distsum) : 0.0f;

						filtervalue0 += this_weight*noisy_value_searchpos0;
						filtervalue1 += this_weight*noisy_value_searchpos1;
						filtervalue2 += this_weight*noisy_value_searchpos2;

						filterweight0 += this_weight;

						if (this_weight > maxweight0) maxweight0 = this_weight;
					}
					else
					{
						float this_weight0 = (distance0 > expapproximation_cutoff) ? expapproximation(distance0) : 0.0f;
						float this_weight1 = (distance1 > expapproximation_cutoff) ? expapproximation(distance1) : 0.0f;
						float this_weight2 = (distance2 > expapproximation_cutoff) ? expapproximation(distance2) : 0.0f;

						filtervalue0 += this_weight0*noisy_value_searchpos0;
						filtervalue1 += this_weight1*noisy_value_searchpos1;
						filtervalue2 += this_weight2*noisy_value_searchpos2;

						filterweight0 += this_weight0;
						filterweight1 += this_weight1;
						filterweight2 += this_weight2;

						if (this_weight0 > maxweight0) maxweight0 = this_weight0;
						if (this_weight1 > maxweight1) maxweight1 = this_weight1;
						if (this_weight2 > maxweight2) maxweight2 = this_weight2;
					}
					/////////////////////////////////////////////////////////////////////////

				}
				/////////////////////////////////////////////////////////////////////////

				if (!params->color.independent_channels && maxweight0 > 0.0f)
				{
					filtervalue0 += maxweight0*noisy_value_origin0;
					filtervalue1 += maxweight0*noisy_value_origin1;
					filtervalue2 += maxweight0*noisy_value_origin2;
					filterweight0 += maxweight0;

					results[0][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue0/filterweight0;
					results[1][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue1/filterweight0;
					results[2][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = filtervalue2/filterweight0;
				}
				else if (params->color.independent_channels)
				{
					results[0][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = (maxweight0 > 0.0f) ? filtervalue0/filterweight0 : noisy_value_origin0;
					results[1][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = (maxweight1 > 0.0f) ? filtervalue1/filterweight1 : noisy_value_origin1;
					results[2][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = (maxweight2 > 0.0f) ? filtervalue2/filterweight2 : noisy_value_origin2;
				}
				else
				{
					results[0][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin0;
					results[1][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin1;
					results[2][offset_unpadded + (y0-ypad)*shape[0]+ x0-xpad] = noisy_value_origin2;
				}

				free(values_origin);

				//continue image space
			}
		}

		return;
	}
}

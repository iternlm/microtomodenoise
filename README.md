# iterNLM_v0.3

![alt tag](https://github.com/iternlm/microtomodenoise/blob/master/chalkdemo.png)

This repository provides C++ and CUDA C++ source code and a UNIX build for iterative nonlocal means denoising (iterNLM).
The denoiser is designed to handle correlated noise footprints characteristic to postreconstruction denoising problems of tomography reconstructions. It is based on the works by Antoni Buades et al.:

[1] Buades, A.; Coll, B.; Morel, J.M. *Multiscale Model. Simul.* **2005**, 4(2), 490–530. "A review of image denoising algorithms, with a new one", doi: [10.1137/040616024](http://dx.doi.org/10.1137/040616024).

[2] Coupé, P.; Yger, P.; Prima, S.; Hellier, P.; Kervrann, C.; Barillot, C. *IEEE T. Med. Imaging* **2008**, 27(4), 425–441. "An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images", doi: [10.1109/TMI.2007.906087](http://dx.doi.org/10.1109/TMI.2007.906087).

The iterative implementation of the NLM algorithm enables targeted removal of textures with few redundacies and low contrast-to-noise ratio. Such textures are characteristic to backprojected noise and artefacts in high resolution tomography. A publication providing benchmark results and discussing the quality, implementation and range of application of the iterative NLM procedure has been published in *Advances in Water Resources*. Please consider citing it when you decide to use the algorithm:

[3] Bruns, S.; Stipp, S.L.S.; Sørensen, H.O. *Adv. Water Res.* **2017**, 105, 96-107. "Looking for the Signal: A Guide to Iterative Noise and Artefact Removal in X-ray Tomography Reconstructions of Porous Geomaterials", doi: [10.1016/j.advwatres.2017.04.020](http://dx.doi.org/10.1016/j.advwatres.2017.04.020).

**Changelog**

As of 06.12.2019 the prototype programs nanotomodenoise and microtomodenoise have been replace by iterNLM_v0.3 with the following changes and improvements:
- severly reduced memory requirements
- improved speed in CPU mode by a factor of ~1.5 and added GPU and multiGPU support for further speed improvements (cf. below)
- added individual kernel functions for patch radii of up to 3
- added a countdown timer for expected runtime
- incorporate automated noise estimation and various options for setting the noise and texture level (cf. below)
- added basic support for Poisson noise corrupted 2D RGB TIF images in CPU mode

![alt tag](https://github.com/iternlm/microtomodenoise/blob/master/iterNLM-Benchmark2.png)

**Compilation**

Required libraries are LibTiff and OpenMP. The source code should compile on most Linux distributions by providing the location of your nvcc compiler and the CUDA compute capability of your GPU in the script file *make_iternlm.sh*. Without the latter set the compute capability to 0. Execute with *sh make_iternlm.sh* which will provide an executable *iterNLM* in the same directory.

**Usage**

The currently available implementation is limited to 8 bit, 16 bit, 32 bit greyscale TIF image sequences and spatially uniform noise levels in XY. With default settings the noise level is evaluated with a z-adaptive moving window. ImageJ 3D tifs are supported but require the *--blocks* option in CPU mode. We rarely use 2D, RGB or denoising of spatially varying noise, i.e. the code is less maintained.

It is not necessary to set the noise level manually but when using a manual noise estimate the noise level (s0) and texture level (s1) need to be provided manually *before* and *after* the first denoising iteration. The easiest way to do this is to select (expected) uniform regions in the image and measure the standard deviation several times. Use a lower boundary estimate and run a single iteration of denoising before repeating the procedure to estimate s1.

The program is best run from the command line by calling *iterNLM* with the following program arguments:

**Basic Arguments**

| argument | value | explanation |
|--------|------------------|-----------|
| **-i** |/directory/with/noisy/images/| (*mandatory*) needs to contain a minimum of 3 images|
| **-o** |/output/directory/| (*optional*) default output is 1 level above input directory. Separate outputs are generated for each iteration.|
| **-a**|0 > float >= 1|(*optional*, default=0.5) weighting parameter alpha for s0 and s1. Lower alphas preserve more detail. Higher alphas impose stronger artefact removal.|
| **-noise** |string| (*optional*, default=z-adaptive) determines a mode for how the noise level is estimated (cf. below)|
| **-search** |integer| (*optional*) set all dimensions of the search space to a uniform radius. 5 may be preferable when there is only backprojected noise and little artefacts.|
|**--cleanup**|| (*optional*) only keep the final denoising result on the HDD. By default every iteration is exported.|

The straightforward program call would thus be: *./iterNLM -i /directory/with/noisy/images/*

Available modes for setting the noise and texture level (*-noise*):

- *global* (calculates the variance in 2D patches and separates the result in two clusters. The lower value is the noise estimate.
- *z-adaptive* (The same as above but instead of a uniform noise level a moving window of 100 slices is considered. Especially useful for zoom-tomography in irregular shaped samples
- *semimanual* (Provide the noise level s0. A patch with a similar variance is selected and used to estimate the texture level.)
- *manual* (Provide the noise level s0 and the texture level s1 manually.)

**Denoiser Related Arguments**

| argument | value | explanation |
|--------|------------------|-----------|
| **-iter**|integer|(*optional*, default=4) number of denoising iterations|
| **-a**|0 > float >= 1|(*optional*, default=0.5) weighting parameter alpha for s0 and s1. Lower alphas preserve more detail. Higher alphas impose stronger artefact removal.|
|**-slices**|integer|(*optional*, default=11) amount of 3D information used for denoising the central slice. Default settings are high quality requiring plenty of resources. 7 or 5 are usually sufficient for avoiding artefacts in z-direction. Try reverting to 3 slices on older machines.|
| **-search0** |integer| (*optional*, default=10) radius of the search space in the first dimension|
| **-search1** |integer| (*optional*, default=10) radius of the search space in the second dimension|
| **-search2** |integer| (*optional*, default=10) radius of the search space in the third dimension|
| **-search** |integer| (*optional*) set all dimensions of the search space to a uniform radius. 5 may be preferable when there is only backprojected noise and little artefacts.|
| **-patch0** |integer| (*optional*, default=1) radius of the patch space in the first dimension|
| **-patch1** |integer| (*optional*, default=1) radius of the patch space in the second dimension|
| **-patch2** |integer| (*optional*, default=1) radius of the patch space in the third dimension|
| **-patch** |integer| (*optional*) set all dimensions to a uniform radius. For soft tissue images a value of 2 or 3 may (or may not) be more suited|

**Noise Level Related Arguments**

| argument | value | explanation |
|--------|------------------|-----------|
| **-noise** |string| (*optional*, default=z-adaptive) determines a mode for how the noise level is estimated (cf. above)|
|**-noiseshift**|float|(*optional*, default=1) shift the estimated noise level by noiseshift times the standard deviation of the noise estimate|
| **-s** |float| (*optional*) set both noise levels at once. Equivalent to alpha=1|
| **-s0**|float|(*optional*) allows setting the noise level manually|
| **-s1**|float|(*optional*) allows setting the texture level manually = noise standard deviation after 1 denoising iteration|
| **-nsamples**|integer|(*optional*, default=1e5) number of samples drawn for estimating the noise level|
|**-noisepatch**|integer|(*optional*, default=15) size of 2D window used for automatic noise estimation|
|**--continuous**||update the noise estimate after every iteration and not only the first two (not recommended)|

**Hardware Related Arguments**

| argument | value | explanation |
|--------|------------------|-----------|
|**-cpus**|integer| (*optional*, default=128) Provides an upper limit to the amount of threads used by the CPU.|
|**-gpus**|integer| (*optional*, default=2) Provides an upper limit to the amount of GPUs used by the denoiser.|
|**-gpu0**|integer| (*optional*, default=0) device ID of the first GPU used by the denoiser. Use this when you have multiple GPUs and GPU0 is already occupied.|
|**--noblocks**|| (*optional*) keep everything in host RAM. This is default in GPU mode but may speed-up CPU mode. By default active slices are grabbed from the HDD when required to save on memory requirements.|
|**--blocks**|| (*optional*) When low on host memory this should help. Forces the program to only read into memory what is currently needed.|
|**--v**|| (*optional*) verbose mode currently only provides an estimate of the memory requirement in CPU mode. Might be helpful if you encounter crashes.|
|**-threads**|integer| (*optional*, default=128) sets threadsPerBlock in CUDA. No reason to touch this.|

**Arguments Related to 2D RGB Denoising Only**

| argument | value | explanation |
|--------|------------------|-----------|
|**--color**||activates the RGB denoiser|
|**--nopoisson**||when images are corrupted with Gaussian noise only|
|**--noaverage**||by default the noise level is averaged across all channels. Keep an individual estimate for each channel with this option|
|**--independent**||by default image similarity is measured across R,G and B channel. Choose this option for denoising them independently|

**Further Optional Arguments**

| argument | value | explanation |
|--------|------------------|-----------|
| **--resume**|| (*optional*) tries resuming a previously interrupted denoising job|
| **--16bit**|| (*optional*) allows safing disk space by only generating 16bit results|
| **-fs**| integer| (*optional*) subset denoising, first slice|
| **-ls**| integer| (*optional*) subset denoising, last slice|

Please contact bruns@nano.ku.dk or osholm@nano.ku.dk for assistance, requests or recommendations.

Applications in science:
* Nielsen,M. S.; Munk, M. B.; Diaz, A.; Pedersen, E. B. L.; Holler, M.; Bruns, S.; Risbo, J.; Mortensen, K.; Feidenhans’l, R. K. *Food Structure* **2016**, 7, 21–28. “Ptychographic X-ray Computed Tomography of Extended Colloidal Networks in Food Emulsions”, doi: [10.1016/j.foostr.2016.01.001](http://dx.doi.org/10.1016/j.foostr.2016.01.001).
* Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. *Adv. Water. Res.* **2017**, 107, 32-42. “Statistical Representative Elementary Volumes of Porous Media determined using Greyscale Analysis of 3D Tomograms”, doi: [10.1016/j.advwatres.2017.06.002](https://doi.org/10.1016/j.advwatres.2017.06.002).
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. *Environ. Sci. Technol.* **2017**, 51(14), 7982-7991, “Dissolved CO 2 stabilizes Dissolution Front and increases Breakthrough Porosity of Natural Porous Materials”, doi: [10.1021/acs.est.7b02157](https://doi.org/10.1021/acs.est.7b02157).
* Chavez Panduro, E. A.; Torsæter, M.; Gawel, K.; Bjørge, R.; Gibaud, A.; Yang, Y.; Bruns, S.; Zheng, Y.; Sørensen, H. O.; Breiby, D. *Environ. Sci. Technol.* **2017**, 51(16), 9344-9351. “In-situ X-ray Tomography Study of Cement exposed to CO 2 Saturated Brine“, doi: [10.1021/acs.est.6b06534](https://doi.org/10.1021/acs.est.6b06534).
* Yousefi, N.; Wong, K.; Hosseinidoust, Z.; Sørensen, H.O.; Bruns, S.; Zheng, Y.; Tufenkji, N *Nanoscale* **2018**, 10(15), 7171-7184. "Ultra-strong graphene oxide-cellulose nanocrystal nanohybrid sponges for efficient water treatment", doi: [10.1039/c7nr09037d](https://doi.org/10.1039/c7nr09037d).
* Yang, Y.; Bruns, S.; Rogowska, M.; Hakim, S. S.; Hammel, J.; Stipp, S. L. S.; Sørensen, H. O. *Sci. Rep.* **2018**, 8, 5693. “Retraction of the Dissolution Front in Natural Porous Media”, doi: [10.1038/s41598-018-23823-3](https://doi.org/10.1038/s41598-018-23823-3).
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. *Adv. Water. Res.* **2018**, 115, 151-159. “Impact of microstructure evolution on the difference between geometric and reactive surface areas in natural chalk”., doi: [10.1016/j.advwatres.2018.03.005](https://doi.org/10.1016/j.advwatres.2018.03.005).
* Yang, Y.; Hakim, S. S.; Bruns, S.; Rogowska, M.; Böhnert, S.; Hammel, J.; Stipp, S. L. S.; Sørensen, H. O. *ACS Earth Space Chem.* **2018**, 2(6), 618-633. “Direct Observation of Coupled Geochemical and Geomechanical Impacts on Chalk Microstructure Evolution under Elevated CO2 Pressure”, doi: [10.1021/acsearthspacechem.8b00013](https://doi.org/10.1021/acsearthspacechem.8b00013).
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. *PLoS One* **2018**. “Patterns of entropy production in dissolving natural porous media with flowing fluid”, doi: [10.1371/journal.pone.0204165](https://doi.org/10.1371/journal.pone.0204165).
* Yang, Y.; Rogowska, M.; Zheng, Y.; Bruns, S.; Gundlach, C.; Stipp, S. L. S.; Sørensen, H. O. *J. Hydrol.* **2019**, 571, 21-35. “Transient increase in reactive surface and the macroscopic Damkohler number in chalk dissolution”, doi: [10.1016/j.jhydrol.2019.01.032](https://doi.org/10.1016/j.jhydrol.2019.01.032).
* Yang, Y.; Hakim, S. S.; Bruns, S.; Uesugi, K.; Stipp, S. L. S.; Sørensen, H. O. *Water Resour. Res.* **2019**, 55(6), 4801-4819. “Effect of Cumulative Surface on Pore Development in Chalk”, doi: [10.1029/2018WR023756](https://doi.org/10.1029/2018WR023756).



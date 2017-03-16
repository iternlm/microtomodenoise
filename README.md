# microtomodenoise

![alt tag](https://github.com/iternlm/microtomodenoise/blob/master/chalkdemo.png)

This repository provides C++ source code (as a CodeBlocks project) and builds on Ubuntu 14.04 and 16.04 for iterative nonlocal means denoising (iterNLM).
The denoiser is designed to handle correlated noise footprints characteristic to postreconstruction denoising problems of tomography reconstructions. It is based on the works by Antoni Buades et al.:

[1] Buades, A.; Coll, B.; Morel, J.M. *Multiscale Model. Simul.* **2005**, 4(2), 490–530. "A review of image denoising algorithms, with a new one", doi: [10.1137/040616024](http://dx.doi.org/10.1137/040616024).

[2] Coupé, P.; Yger, P.; Prima, S.; Hellier, P.; Kervrann, C.; Barillot, C. *IEEE T. Med. Imaging* **2008**, 27(4), 425–441. "An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images", doi: [10.1109/TMI.2007.906087](http://dx.doi.org/10.1109/TMI.2007.906087).

The iterative implementation of the NLM algorithm enables targeted removal of textures with few redundacies and low contrast-to-noise ratio. Such textures are characteristic to backproject noise and artefacts in high resolution tomography. A publication providing benchmark results and discussing the quality, implementation and range of application of the iterative NLM procedure is currently being reviewed. Please consider citing it when you decide to use the algorithm:

[3] Bruns, S.; Stipp, S.L.S.; Sørensen, H.O. *Adv. Water Res.* **2016**, *under review*. "Looking for the Signal: A Guide to Iterative Noise and Artefact Removal in X-ray Tomography Reconstructions of Porous Geomaterials".

The currently available implementation is limited to 8 bit, 16 bit, 32 bit greyscale image sequences and spatially uniform noise levels. We rarely use 2D, RGB or denoising of spatially varying noise, i.e. the code is less maintained.

The noise level (s0 and s1) needs to be provided manually *before* and *after* the first denoising iteration. The easiest way to do this is to select (expected) uniform regions in the image and measure the standard deviation several times. Use a lower boundary estimate and run a single iteration of denoising before repeating the procedure to estimate s1.

The program is best run from the command line by calling *microtomodenoise* with the following program arguments:

| argument | value | explanation |
|--------|------------------|-----------|
| **-i** |/directory/with/noisy/images/| (*mandatory*) needs to contain a minimum of 3 images|
| **-o** |/output/directory/| (*optional*) default output is 1 level above input directory. Separate outputs are generated for each iteration.|
| **-s0**|double|(*mandatory*) initial noise standard deviation|
| **-s1**|double|(*mandatory*) texture level = noise standard deviation after 1 denoising iteration|
| **-iter**|integer|(*optional*, default=4) number of denoising iterations|
| **-a**|0 > double >= 1|(*optional*, default=0.5) weighting parameter alpha for s0 and s1. Lower alphas preserve more detail. Higher alphas impose stronger artefact removal.|
|**-slices**|integer|(*optional*, default=11) amount of 3D information used for denoising the central slice. Default settings are high quality requiring plenty of resources. Try reverting to 3 slices on older machines.|

Further optional arguments:

| argument | value | explanation |
|--------|------------------|-----------|
| **-search0** |integer| (*optional*, default=10) radius of the search space in the first dimension|
| **-search1** |integer| (*optional*, default=10) radius of the search space in the second dimension|
| **-search2** |integer| (*optional*, default=5) radius of the search space in the third dimension|
| **-r**|boolean|(*optional*, default=true) resumes the denoising operation. Set *false* to overwrite previously generateed results with different parameter settings.|
| **--16bit**| | (*optional*) allows safing disk space by only generating 16bit results|
| **-s**| double | set both noise levels at once. Equivalent to alpha=1|
| **-fs**| integer| (*optional*) subset denoising, first slice|
| **-ls**| integer| (*optional*) subset denoising, last slice|

Please contact bruns@nano.ku.dk or osholm@nano.ku.dk for assistance, requests or recommendations.

Applications in science:
* Nielsen,M. S.; Munk, M. B.; Diaz, A.; Pedersen, E. B. L.; Holler, M.; Bruns, S.; Risbo, J.; Mortensen, K.; Feidenhans’l, R. K. *Food Structure* **2016**, 7, 21–28. “Ptychographic X-ray Computed Tomography of Extended Colloidal Networks in Food Emulsions”, doi: [10.1016/j.foostr.2016.01.001](http://dx.doi.org/10.1016/j.foostr.2016.01.001).
* Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. *Adv. Water. Res.* **2017**, *under review*. “Properties of Porous Media determined using Greyscale Analysis of 3D Tomograms”.
* Chavez Panduro, E. A.; Torsæter, M.; Gawel, K.; Bjørge, R.; Gibaud, A.; Yang, Y.; Bruns, S.; Zheng, Y.; Sørensen, H. O.; Breiby, D. *Environ. Sci. Technol.* **2017**, *under review*. “In-situ X-ray Tomography Study of Cement exposed to CO 2 Saturated Brine“.
* Yousefi, N.; Wong, K.; Hosseinidoust, Z.; Sørensen, H.O.; Bruns, S.; Zheng, Y.; Tufenkji, N *Nat. Commun.* **2017** *submitted*. "Ultra-strong graphene oxide-cellulose nanocrystal nanohybrid sponges for efficient water treatment".

In preparation:
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Patterns of Entropy Production in the Microstructural Development of North Sea Chalk”.
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Reactive Infiltration Instability amplifies the Difference between Geometric and Reactive Surface Areas in Natural Porous Materials”.
* Yang, Y.; Bruns, S.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Dissolved CO 2 stabilizes Dissolution Front and increases Breakthrough Porosity of Natural Porous Materials”.
* Yang, Y.; Hakim, S. S.; Bruns, S.; Uesugi, K.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Wormholes grow along paths with Minimal Cumulative Surface”.
* Yang, Y.; Hakim, S. S.; Bruns, S.; Rogowska, M.; Böhnert, S.; Hammel, J.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Direct Observation of coupled Geochemical and Geomechanical Impacts on Chalk Microstructural Evolution under elevated CO 2 pressure. Part I”.
* Yang, Y.; Hakim, S. S.; Bruns, S.; Rogowska, M.; Hammel, J.; Stipp, S. L. S.; Sørensen, H. O. **2017**. “Direct Observation of coupled Geochemical and Geomechanical Impacts on Chalk Microstructural Evolution under elevated CO 2 pressure. Part II”.

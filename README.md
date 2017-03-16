# microtomodenoise

![alt tag](https://github.com/iternlm/microtomodenoise/blob/master/chalkdemo.png)

This repository provides C++ source code (as a CodeBlocks project) and builds on Ubuntu 14.04 and 16.04 for iterative nonlocal means denoising (iterNLM).
The denoiser is designed to handle correlated noise footprints characteristic to postreconstruction denoising problems of tomography reconstructions. It is based on the works by Antoni Buades et al.:

[1] Buades, A., Coll, B., and Morel, J.M. (**2005**) "A review of image denoising algorithms, with a new one", *Multiscale Model. Simul.*, 4(2), 490–530, doi: [10.1137/040616024](http://dx.doi.org/10.1137/040616024).

[2] Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., and Barillot, C. (**2008**), "An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images", *IEEE T. Med. Imaging*, 27(4), 425–441, doi: [10.1109/TMI.2007.906087](http://dx.doi.org/10.1109/TMI.2007.906087).

The iterative implementation of the NLM algorithm enables a targeted removal of textures with few redundacies and low contrast-to-noise ratio. Such textures are characteristic to backproject noise and artefacts in high resolution tomography. A publication discussing the quality, implementation and range of application of the iterative NLM procedure is currently being reviewed. Please consider citing it when you decide to use the algorithm:

[3] Bruns, S., Stipp, S.L.S., Sørensen, H.O. (**2016**) "Looking for the Signal: A Guide to Iterative Noise and Artefact Removal in X-ray Tomography Reconstructions of Porous Geomaterials", *Adv. Water Res.*, submitted.

  

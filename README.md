Denoising and Inpainting of Images Using Wavelets
=================================================

The code in this project was a part of my final submission for the class YSC4206 Harmonic Analysis at Yale-NUS College. I use [B-Spline wavelets](https://en.wikipedia.org/wiki/Spline_wavelet) for the denoising and inpainting of "natural" images i.e. images that are not themselves pictures of noise or randomly generated pixels. 

The basic idea is that real-world images have large smooth regions. It is well-known that sparse wavelet coefficients characterise the regularity of functions, so real-world images are sparse in the wavelet domain. 

Noisy and/or corrupted images are linearly-distorted versions of the original image with a noise constant added in. The recovery of sparse signals from noisy, linearly-distorted signals is a well-studied topic; see [this paper](https://arxiv.org/abs/1012.0621) by Recht et al. The problem is reduced to a cost-minimisation problem, with the cost function given by the L2 distance to the original image with an added regularisation term that is the L1 norm in the wavelet domain. 

See the file 'examples.py' for usage and example output. 

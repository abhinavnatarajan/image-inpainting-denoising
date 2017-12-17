import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

from wavelet_inpaint_denoise.apgd import apgd
import wavelet_inpaint_denoise.core

#image we already have
img = imread('Example images/input1.png')/255
A = 1-(img==1.0).astype(np.float64)
img*=A
img1=apgd(img, A)
imsave('Example images/output.png', img1)

#load the cameraman image
img = imread('Example images/cameraman.png')/255


#add some noise
noise_scale = 0.1
img_noisy = img + noise_scale*np.random.normal(size=img.shape)
imsave('Example images/cameraman_with_noise.png', img_noisy)


img1 = apgd(img_noisy, np.ones(img.shape))
imsave('Example images/cameraman_with_noise_corrected.png', img1)


#delete random parts
A = np.random.randint(0,2,size=img.shape)
img_corr = img * A
imsave('Example images/cameraman_corrupted.png', img_corr)


img2 = apgd(img_corr, A)
imsave('Example images/cameraman_corrupted_corrected.png', img2)


#add noise to corrupted image
img_corr_noisy = img_corr + noise_scale*np.random.normal(size=img.shape)
imsave('Example images/cameraman_noisy_corrupted.png', img_corr_noisy)

img3 = apgd(img_corr_noisy, A)
imsave('Example images/cameraman_noisy_corrupted_corrected.png', img3)

img1 = imread('Example images/cameraman.png')/255
wc = wavelet_inpaint_denoise.core.dwt2(img1, haar, 5)
for i in range(1,5):
    for j in range(2):
        for k in range(2):
            plt.figure()
            plt.imshow(wc[i,j,k])
plt.show()

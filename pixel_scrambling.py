#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:30:42 2025

@author: sydney
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
import skan
from skan.csr import skeleton_to_csgraph, pixel_graph
import matplotlib.pyplot as plt
from skan import Skeleton, summarize
from skan import csr

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure  # Import exposure module from skimage
from skimage import morphology
import imageio.v3 as iio
from skan.pre import threshold
from skan import draw
from skimage import exposure

height, width = None, None

# Compute the Power Spectrum of the Skeletonized Image
def compute_power_spectrum(skeleton):
    # Compute the 2D FFT of the skeleton image
    fft_image = np.fft.fft2(skeleton)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift zero frequency to center
    power_spectrum = np.abs(fft_shifted) ** 2  # Compute power spectrum

    # Normalize and log-transform for better visualization
    power_spectrum_log = np.log1p(power_spectrum)  # log(1 + x) to avoid log(0)
    
    return power_spectrum_log

def radial_average(power_spectrum):
    # Get the dimensions of the power spectrum
    ny, nx = power_spectrum.shape
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    X, Y = np.meshgrid(x, y)
    
    # Compute the radial distance from the center
    r = np.sqrt(X**2 + Y**2)
    r = r.astype(np.int32)  # Convert to integer for binning

    # Get unique radii
    r_max = np.max(r)
    radial_mean = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)

    # Compute sum and count per radius
    for i in range(ny):
        for j in range(nx):
            radial_mean[r[i, j]] += power_spectrum[i, j]
            counts[r[i, j]] += 1

    # Normalize to get the mean
    radial_mean /= np.where(counts == 0, 1, counts)

    return radial_mean



# Load image

img = iio.imread('M6765_3610 YFP_chaotic pattern (1) (1).tif')[170]
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 # Convert from BGR to RGB

# take the power spectrum of original image
power_spectrum = compute_power_spectrum(img)
radial_profile = radial_average(power_spectrum)
radial_profiles = [] 

rep=10 # number of replicates
total_power_spectrum = None
for i in range(rep):
    

    # Flatten the image to 1D and shuffle
    h, w = img.shape
    pixels = img.flatten()
    np.random.shuffle(pixels) 
    
    # Reshape back to original shape
    scrambled = pixels.reshape(h, w)
    
    # take the power spectrum of scrambled image
    scram = compute_power_spectrum(scrambled)
    radial_profile_scram = radial_average(scram)
    
    
    # Store the radial profile for this replicate
    radial_profiles.append(radial_profile_scram)


# Convert list of radial profiles to a NumPy array for easy calculation
radial_profiles = np.array(radial_profiles)

# Compute the mean and standard deviation along the replicate axis (axis=0)
mean_power_spectrum = radial_profiles.mean(axis=0)
std_dev_power_spectrum = radial_profiles.std(axis=0)

# Display
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Display
plt.imshow(scrambled)
plt.title('Scrambled Image')
plt.axis('off')
plt.show()

# Plot the radial average of a single frame
plt.figure()
plt.plot(radial_profile, label="Original")
plt.plot(mean_power_spectrum, label="Scrambled average")
plt.xlabel("Radius")
plt.ylabel("Power Spectrum Average")
plt.title("Radial Average of Power Spectrum")
#plt.xlim(-50,550)
plt.yscale("log")
plt.legend()
plt.show()

# Plot the radial average of a single frame with error bars for standard deviation
plt.figure()
plt.plot(radial_profile, label="Original")
plt.errorbar(np.arange(len(mean_power_spectrum)), mean_power_spectrum, yerr=std_dev_power_spectrum, label="Scrambled average",  capsize=3)
plt.xlabel("Radius")
plt.ylabel("Power Spectrum Average")
plt.title("Radial Average of Power Spectrum with Standard Deviation")
plt.yscale("log")
plt.legend()
plt.show()


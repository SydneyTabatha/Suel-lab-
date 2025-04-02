#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:45:05 2025

@author: sydney
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
from scipy.ndimage import convolve
'''
def load_skeleton_movie(file_path):
    #Load the skeletonized movie as a 3D numpy array (frames, height, width)."""
    return io.imread(file_path)


def count_trifurcations(movie):
    #Count trifurcations (skeleton pixels with exactly 3 neighbors) in each frame."""
    trifurcation_counts = []
    
    # Define a 3x3 kernel to count neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    for frame in movie:
        skeleton = (frame > 0).astype(np.uint8)  # Binarize the skeleton
        neighbors = convolve(skeleton, kernel, mode='constant', cval=0)  # Count neighbors
        
        # A trifurcation occurs where a skeleton pixel has exactly 4 neighbors (including itself)
        trifurcations = (skeleton == 1) & (neighbors == 4)
        trifurcation_counts.append(np.sum(trifurcations))
    
    return np.array(trifurcation_counts) 

def plot_trifurcation_counts(trifurcation_counts):
    #Plot the number of trifurcations as a function of time (frames)."""
    plt.figure(figsize=(8, 6))
    plt.plot(trifurcation_counts, marker='o', linestyle='-')
    plt.xlabel("Frame")
    plt.ylabel("Number of Trifurcations")
    plt.title("Trifurcation Count Over Time")
    plt.show()

if __name__ == "__main__":
    # ---- Step 1: Load the skeleton movie ----
    file_path = '/Users/sydney/Desktop/Suel lab rotation/processed_movie_with_display_settings.tif'
    movie = load_skeleton_movie(file_path)

    # ---- Step 2: Count trifurcations ----
    trifurcation_counts = count_trifurcations(movie)

    # ---- Step 3: Plot results ----
    plot_trifurcation_counts(trifurcation_counts)
    
    
# -------------------------------------------------|

#y = [10,  2,  2,  7, 10,  6,  9, 12,  6,  7,  8, 10,  9,  2,  4,  7,  6, 11,  8,  2,  5,  4, 11,  9,  4,  8,  4,  6,  7,  2,  5, 10,  6,  4, 5,  8,  9,  9, 11,  8,  6,  5,  6,  6,  4,  5,  4,  3,  5,  7, 13, 12,  8,  7,  7,  8,  7,  3,  6,  4,  6,  4,  6,  5,  9,  7,  8,  6, 4,  9,  7,  8,  4,  7, 12, 10,  5,  9, 12,  8,  6,  7,  6,  8, 12, 3,  3,  8,  8,  7,  7,  6,  8, 10, 18,  7, 10, 12,  7, 11, 11,  4, 10,  4,  6,  6,  6,  2,  6,  7,  7, 10,  6,  5,  6,  4,  8,  6,  6, 2,  6, 10, 11, 10,  7,  3,  6,  7,  6, 19, 10,  5, 10,  5,  4,  8, 6, 12,  7,  7,  8,  7, 10,  5,  7,  6, 12,  9, 12,  6, 12,  4,  4, 4,  6,  9,  4,  5,  8,  7,  9,  5, 11, 11,  7,  8,  8, 11,  8,  6, 9,  8,  8, 10, 10,  6, 11,  9,  5];



'''







#trifurcation_counts = [17, 11, 9, 10, 8, 12, 10, 8, 11, 7, 6, 6, 7, 10, 6, 8, 4, 3, 4, 9, 6, 5, 6, 5, 5, 8, 5, 7, 6, 10, 8, 14, 6, 12, 9, 11, 9, 11, 9, 8, 9, 11, 7, 6, 5, 12, 9, 10, 17, 9, 18, 8, 11, 14, 11, 9, 10, 13, 9, 10, 10, 17, 14, 7, 13, 13, 13, 13, 8, 13, 12, 14, 8, 12, 12, 9, 13, 8, 10, 6, 8, 13, 6, 9, 12, 8, 9, 9, 9, 12, 13, 11, 11, 10, 12, 10, 12, 12, 11, 8, 12, 9, 7, 11, 10, 9, 8, 9, 11, 16, 13, 8, 15, 9, 8, 8, 9, 12, 12, 9, 9, 13, 11, 8, 13, 13, 13, 16, 12, 18, 11, 8, 11, 10, 12, 12, 12, 13, 6, 11, 12, 8, 12, 10, 9, 11, 10, 14, 16, 10, 12, 10, 9, 8, 7, 11, 12, 13, 9, 12, 7, 15, 14, 15, 8, 9, 7, 13, 6, 8, 14, 7, 7, 8, 7, 8, 12, 9]
trifurcation_counts = [12, 10, 11, 8, 13, 12, 8, 9, 10, 9, 11, 7, 11, 10, 7, 8, 11, 8, 11, 10, 8, 8, 6, 2, 7, 3, 4, 8, 7, 3, 8, 9, 5, 5, 5, 7, 4, 9, 6, 11, 9, 5, 8, 8, 6, 6, 11, 9, 10, 15, 13, 19, 10, 11, 9, 8, 9, 10, 6, 10, 12, 9, 7, 11, 9, 8, 8, 12, 14, 11, 10, 11, 8, 14, 13, 13, 10, 13, 14, 9, 10, 8, 5, 7, 9, 12, 14, 8, 9, 9, 10, 10, 10, 10, 13, 9, 12, 9, 9, 12, 6, 11, 10, 15, 10, 8, 8, 9, 10, 10, 7, 7, 13, 10, 10, 8, 10, 12, 11, 14, 9, 10, 10, 8, 7, 9, 11, 10, 9, 11, 12, 10, 12, 7, 8, 12, 8, 13, 6, 10, 8, 11, 16, 11, 14, 11, 11, 10, 7, 7, 8, 11, 5, 7, 8, 15, 5, 7, 11, 11, 9, 7, 11, 6, 14, 11, 10, 11, 9, 10]


plt.figure(figsize=(8, 6))
plt.plot(trifurcation_counts, marker='o', linestyle='-')
plt.xlabel("Frame")
plt.ylabel("Number of Trifurcations")
plt.title("Trifurcation Count Over Time")
plt.show()


# attractor reconstruction
# -------------------------------------------------|
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def reconstruct_phase_space(time_series, delay, dimension):
    """
    Perform time-delay embedding to reconstruct the phase space.
    
    Parameters:
        time_series (array): 1D array of time series data (e.g., number of trifurcations).
        delay (int): Time delay τ.
        dimension (int): Embedding dimension d.
    
    Returns:
        embedded_data (2D array): Reconstructed phase space.
    """
    N = len(time_series)
    embedded_data = np.array([time_series[i: N - delay * (dimension - 1) + i: delay] for i in range(dimension)]).T
    return embedded_data

# ---- Load your time series data ----
# Replace this with your actual data
time_series = trifurcation_counts # Load from a text file

# ---- Set embedding parameters ----
delay = 5        # Time delay τ (adjust based on autocorrelation or mutual information)
dimension = 2    # Embedding dimension d (adjust based on False Nearest Neighbors method)

# ---- Reconstruct the phase space ----
embedded_data = reconstruct_phase_space(time_series, delay, dimension)

# ---- Plot the reconstructed phase space ----
plt.figure(figsize=(8, 6))
plt.plot(embedded_data[:, 0], embedded_data[:, 1], lw=0.5)
plt.title("Reconstructed Phase Space (2D)")
plt.xlabel("X(t)")
plt.ylabel(f"X(t + {delay})")
plt.show()



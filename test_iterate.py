#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:58:05 2025

@author: sydney
"""
# CURRENTLY WORKING ON !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# show first 10 frames of skeleton from other method
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
import skan
from skan.csr import skeleton_to_csgraph, pixel_graph
import matplotlib.pyplot as plt
from skan import Skeleton, summarize


import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure  # Import exposure module from skimage
from skimage import morphology
import imageio.v3 as iio
from skan.pre import threshold
from skan import draw
from skimage import exposure


'''
Documentation
https://skeleton-analysis.org/stable/getting_started/getting_started.html#extracting-a-skeleton-from-an-image
'''



'''
def load_skeleton_movie(file_path):
    """Load the skeletonized movie as a 3D numpy array (frames, height, width)."""
    cap = cv2.VideoCapture(file_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if the frame is in color
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()
    return np.array(frames)

'''



# script ---------------------------------------------------|

#file_path = '/Users/sydney/Desktop/Suel lab rotation/output_movie.mp4'
#video_frames = load_skeleton_movie(file_path)

height, width = None, None
frames = []
trifurcations = []
for i in range(178):
    #image0 = iio.imread('M6688_Stack_G003_only.tif')[i]  # Example: using the first frame from the stack
    #image0 = iio.imread('M6800 BS3610 pH=6.tif')[i] 
    image0 = iio.imread('M6800 BS3610 pH=6.tif')[i] 
    # Invert the image to focus on the darkest regions (darkest become white)
    image0_inverted = np.invert(image0)
    image0_norm = exposure.rescale_intensity(image0_inverted, in_range='image', out_range=(0, 1))
    # Define the pixel size (spacing_nm) - adjust based on your image's resolution
    spacing_nm = 0.1  # Example: 100 nm per pixel (replace with actual value)
    # still need to ask about pixel size!
    
    
    # Preprocess the selected frame (smoothing and thresholding)
    smooth_radius = 5 / spacing_nm  # Smooth over a region of 5 times the pixel size
    threshold_radius = int(np.ceil(50 / spacing_nm))  # Threshold size of ~50 nm
    binary0 = threshold(image0_norm, sigma=smooth_radius, radius=threshold_radius)
    
    # Apply skeletonization to the binary image
    skeleton0 = morphology.skeletonize(binary0)
    
    # Increase the brightness of the original image for plotting
    image0_bright = np.clip(image0 * 2, 0, 255)  # Increase brightness by multiplying by a factor (e.g., 1.5)

    # Apply histogram equalization
    image0_contrast = exposure.equalize_hist(image0)
    '''
    # Now overlay the skeleton
    fig, ax = plt.subplots()
    ax.imshow(image0_bright, cmap='gray')  # Brightened image
    ax.imshow(skeleton0, cmap='jet', alpha=0.5)  # Overlay skeleton in a different color with transparency
    ax.axis('off')
    plt.show()
    '''
    '''
    # Create the figure
    fig, ax = plt.subplots()
    ax.imshow(image0_bright, cmap='gray')
    ax.imshow(skeleton0, cmap='jet', alpha=0.5)
    ax.axis('off')

    # Save the frame
    frame_path = f"frame_{i}.png"
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Read the saved frame and store it
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    frames.append(frame)
    '''
    
    # Create figure in memory
    fig, ax = plt.subplots()
    ax.imshow(image0_bright, cmap='gray')
    ax.imshow(skeleton0, cmap='jet', alpha=0.5)
    ax.axis('off')
    
    # Convert figure to numpy array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())  # Capture as RGBA
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert to BGR for OpenCV
    
    # Store frame for video
    if height is None or width is None:
        height, width, _ = frame.shape
    frames.append(frame)
    
    plt.close(fig)  # Free memory



    
    # Get summary stats
    pixel_graph, coordinates = skeleton_to_csgraph(skeleton0)
    branch_data = summarize(Skeleton(skeleton0, spacing=spacing_nm), separator='_')
    branch_data.head()
    
    # Extract junction coordinates
    junctions = branch_data[branch_data['branch_type'] == 2]
    junction_coords = junctions[['image_coord_src_0', 'image_coord_src_1']].values
    #print(junction_coords)
    # Remove duplicate junctions
    unique_junction_coords = np.unique(junction_coords, axis=0)
    #print(unique_junction_coords)
    # Print the updated count
    print('The total number of unique junctions is', len(unique_junction_coords))
    num_of_nodes = len(unique_junction_coords)
    trifurcations.append(num_of_nodes)
    
print(trifurcations)

# Define the video writer
output_path = "output_video_march31.mp4"
fps = 10  # Adjust frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write frames to the video
for frame in frames:
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as {output_path}")

'''
# Define the video writer
output_path = "output_video_march31.mp4"
fps = 10  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write each frame
for frame in frames:
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as {output_path}")
'''

# to do next

# plug this time sereies of junctions into chaos decision tree algorithm
# read about entropy gurol suggestion

# see if its accurate enough both the skeleton and the number of nodes
# then try with different movies
# then try with different 1-d metrics


# first movie
# [12, 8, 7, 8, 5, 10, 8, 7, 9, 6, 5, 5, 5, 7, 4, 5, 3, 1, 3, 8, 4, 4, 5, 4, 4, 6, 4, 6, 5, 8, 7, 11, 4, 11, 8, 8, 7, 9, 6, 6, 7, 10, 6, 4, 4, 11, 8, 9, 15, 7, 14, 6, 9, 12, 8, 8, 7, 11, 7, 9, 8, 15, 12, 6, 12, 11, 10, 12, 6, 10, 9, 11, 7, 9, 9, 4, 11, 5, 7, 5, 5, 12, 4, 7, 10, 6, 7, 7, 7, 10, 12, 10, 10, 7, 9, 7, 7, 10, 10, 6, 10, 8, 5, 8, 9, 8, 7, 8, 9, 15, 11, 7, 13, 8, 6, 7, 7, 10, 9, 8, 7, 11, 9, 6, 11, 10, 12, 15, 11, 16, 10, 6, 10, 9, 11, 10, 10, 11, 5, 8, 10, 7, 10, 9, 8, 9, 8, 12, 12, 8, 11, 8, 7, 7, 5, 10, 11, 10, 7, 10, 5, 12, 11, 13, 6, 7, 5, 11, 4, 6, 13, 6, 5, 5, 5, 7, 11, 8]
# stochastic from decision tree algorithm

# 2nd movie
# [ 6, 21, 9, 10, 9, 6, 11, 12, 12, 22, 14, 20, 11, 15, 11, 17, 22, 20, 16, 16, 16, 13, 11, 15, 22, 12, 16, 11, 15, 16, 12, 9, 15, 14, 11, 10, 9, 7, 6, 9, 16, 3, 8, 8, 18, 10, 13, 12, 11, 11, 6, 6, 8, 8, 8, 11, 7, 7, 7, 8, 8, 7, 7, 8, 11, 2, 4, 6, 8, 6, 8, 10, 9, 8, 7, 9, 7, 5, 7, 7, 9, 7, 8, 10, 10, 6, 9, 5, 10, 8, 5, 8, 9, 9, 7, 7, 7, 4, 5, 6, 7, 9, 5, 6, 9, 11, 8, 7, 7, 6, 7, 7, 11, 5, 3, 3, 7, 9, 5, 7, 8, 11, 6, 12, 15, 14, 10, 11, 6, 11, 11, 13, 12, 12, 10, 10, 10, 15, 7, 12, 12, 14, 15, 15, 9, 9, 17, 16, 15, 19, 21, 21, 16, 19, 14, 13, 15, 18, 13, 18, 8, 12, 15, 14, 8, 21, 12, 13]
# 47, 37, 35, 40, 35, 27, 32, 18, 21, 17,
# edited out first part
# again stochastic


# third movie
# [38, 32, 33, 32, 17, 27, 16, 12, 9, 9, 10, 6, 11, 10, 6, 7, 9, 7, 9, 4, 10, 7, 5, 6, 9, 7, 9, 8, 6, 7, 4, 1, 6, 1, 3, 7, 6, 2, 6, 5, 3, 4, 3, 5, 2, 6, 5, 10, 7, 4, 7, 6, 4, 3, 9, 8, 9, 14, 10, 18, 9, 9, 7, 6, 8, 8, 4, 8, 11, 8, 5, 9, 8, 6, 7, 11, 13, 10, 9, 10, 6, 10, 11, 11, 8, 12, 13, 8, 9, 7, 4, 6, 6, 11, 12, 6, 8, 8, 9, 9, 9, 9, 12, 8, 11, 8, 7, 10, 3, 9, 7, 14, 9, 7, 7, 8, 9, 8, 6, 5, 11, 9, 8, 7, 9, 11, 9, 12, 8, 9, 9, 7, 6, 7, 10, 8, 8, 9, 11, 9, 10, 6, 7, 11, 5, 11, 5, 8, 7, 10, 15, 7, 11, 10, 8, 9, 5, 6, 6, 10, 3, 6, 6, 11, 4, 5, 9, 8, 8, 5, 10, 5, 12, 9, 9, 10, 7, 8]
# -----------------------------------------------------------|

# to do next 

# 1
# other movies (today) 

# 2
# num of branches (tuesday)
# branch length 
# endpoints
# bin number 

# 3 
# make presentation of results (today)

# 4
# power spectrum analysis
# no filter
# fast fourier transfrom fft
# radial average after 2d fft 
# assuming isotropic 

# 5
# 2d version of entropy analysis? https://pubs.aip.org/aip/cha/article-abstract/33/1/013112/2877516/Multiscale-two-dimensional-permutation-entropy-to?redirectedFrom=fulltext




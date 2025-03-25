#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:12:53 2025

@author: sydney

BEST VERSION SO FAR
"""





import tifffile as tiff
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_ubyte
from skimage.morphology import skeletonize, binary_dilation, closing, disk, remove_small_objects
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_dilation
from skimage.morphology import remove_small_objects
from skimage.measure import label
from PIL import Image
import cv2
import tifffile as tiff
import numpy as np
import cv2
from skimage import exposure
from skimage import img_as_ubyte

# Step 1: Load the TIFF movie
tiff_movie = tiff.imread("M6688_Stack_G003_only.tif")





# rest
# Step 2: Check if frames are loading properly
for i in range(0, len(tiff_movie), max(1, len(tiff_movie) // 10)):  # Show 10 frames evenly spaced
    plt.figure()
    plt.imshow(tiff_movie[i], cmap='gray', vmin=0, vmax=255)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()

# Step 3: ACTUAL PROCESS - Apply Gaussian filter, normalize, skeletonize, and prune
processed_frames = []
smoothed_frames = []
binary_frames = []
sigma = 50  # Standard deviation for Gaussian filter, adjust as needed



# change to 
for frame in tiff_movie:
##for i in range(0, len(tiff_movie), max(1, len(tiff_movie) // 10)):
    
    # then comment this out 
    ##frame = tiff_movie[i]
    
    # Apply Gaussian filter
    smoothed_frame = gaussian_filter(frame, sigma=sigma)
    
    # Normalize the image to the range [0, 1]
    smoothed_frame = (smoothed_frame - smoothed_frame.min()) / (smoothed_frame.max() - smoothed_frame.min())
    
    # Convert to 8-bit for thresholding
    smoothed_frame_8bit = img_as_ubyte(smoothed_frame)
    # INVERT the image so the darkest parts are now the brightest parts
    inverted_frame = 255 - smoothed_frame_8bit
    # Binarize the image
    threshold = 120#155#120 -> not picking up on enough shapes? could try adaptive thresholding 
    binary_frame = inverted_frame > threshold
    
    # a threshold of 120 corresponds to a disk argument of 20
    
    # Apply morphological closing to smooth the blobs
    closed_frame = closing(binary_frame, disk(20))# 55 #20
    
    # Skeletonize the closed binary frame
    skeleton = skeletonize(closed_frame)
    
    # Remove small fragments from the skeleton
    labeled_skeleton = label(skeleton)
    pruned_skeleton = remove_small_objects(labeled_skeleton, min_size=50)
    pruned_skeleton = pruned_skeleton > 0  # Convert back to binary
    
    # Dilate the skeleton for visibility
    skeleton_dilated = binary_dilation(pruned_skeleton)
    
    # Store the processed frame
    binary_frames.append(closed_frame)
    processed_frames.append(skeleton_dilated.astype(np.uint8))
    smoothed_frames.append(smoothed_frame)

# Convert processed frames to a numpy array
processed_movie = np.array(processed_frames)

# Step 4: Save the processed movie as a new TIFF file (if desired)
processed_movie_8bit = [img_as_ubyte(frame) for frame in processed_frames]
output_path = "processed_movie_with_display_settings.tif"
tiff.imwrite(output_path, processed_movie_8bit, bigtiff=True)

# mp4 

# Set video properties (e.g., frame width, height, FPS)
height, width = processed_movie_8bit[0].shape  # Get the dimensions of a single frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
out = cv2.VideoWriter('output_movie.mp4', fourcc, 15.0, (width, height))  # Adjust FPS as needed

# Adjust the contrast (if necessary)
def adjust_contrast(frame, vmin=0, vmax=255):
    """Normalize the frame to the desired contrast range."""
    return np.clip(((frame - frame.min()) / (frame.max() - frame.min())) * (vmax - vmin) + vmin, 0, 255)

# Write each frame to the video with contrast adjustment
for frame in processed_movie_8bit:
    # Adjust the contrast of each frame (use the same contrast settings you used for plotting)
    frame_adjusted = adjust_contrast(frame, vmin=0, vmax=255)  # You can adjust these values as needed
    
    # Convert to BGR (color format) for OpenCV compatibility
    frame_bgr = cv2.cvtColor(frame_adjusted.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Write the adjusted frame to the video
    out.write(frame_bgr)

out.release()  # Finalize and save the video
print("Video saved as output_movie.mp4 with adjusted contrast")



# Step 5: Display processed frames for comparison
for i in range(0, len(smoothed_frames), max(1, len(smoothed_frames) // 10)):
    plt.figure()
    plt.imshow(smoothed_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Smoothed Frame {i}")
    plt.axis('off')
    plt.show()

# Binary frames
for i in range(0, len(binary_frames), max(1, len(binary_frames) // 10)):
    plt.figure()
    plt.imshow(binary_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Binary Frame {i}")
    plt.axis('off')
    plt.show()

# Processed (skeletonized) frames
for i in range(0, len(processed_frames), max(1, len(processed_frames) // 10)):
    plt.figure()
    plt.imshow(processed_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Processed Frame {i}")
    plt.axis('off')
    plt.show()











'''
# Step 1: Load the TIFF movie
tiff_movie = tiff.imread("M6688_Stack_G003_only.tif")

# Step 2: Check if frames are loading properly
# need to change contrast because they are so low with plt.clim()
for i in range(0, len(tiff_movie), max(1, len(tiff_movie) // 10)):  # Show 10 frames evenly spaced
    plt.figure()
    plt.imshow(tiff_movie[i], cmap='gray', vmin=0, vmax=255)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()

# Step 3: ACTUAL PROCESS: Apply Gaussian filter to each frame, normalize and skeletonize
processed_frames = []
smoothed_frames = []
binary_frames = []
sigma = 50#3.0  # Standard deviation for Gaussian filter, adjust as needed
for i in range(0, len(tiff_movie), max(1, len(tiff_movie) // 10)):
    frame = tiff_movie[i]
    # Apply Gaussian filter
    smoothed_frame = gaussian_filter(frame, sigma=sigma)
    
    # Normalize the image to the range [0, 1]
    smoothed_frame = (smoothed_frame - smoothed_frame.min()) / (smoothed_frame.max() - smoothed_frame.min())
    
    ## up untill here it works 
    # Convert to 8-bit (required for skeletonization)
    smoothed_frame_8bit = img_as_ubyte(smoothed_frame)
    # INVERT the image so the darkest parts are now the brightest parts
    inverted_frame = 255 - smoothed_frame_8bit
    # Binarize the image
    threshold = 100
    binary_frame = inverted_frame > threshold
    
    
    # Skeletonize the binary frame
    skeleton = skeletonize(binary_frame)
    
    # Dilate the skeleton to make it more visible
    skeleton_dilated = binary_dilation(skeleton)
    
    # Store the processed frame
    
    binary_frames.append(binary_frame)
    
    processed_frames.append(skeleton_dilated.astype(np.uint8))
    #processed_frames.append(skeleton.astype(np.uint8))  # Convert to uint8 for saving/viewing
    #processed_frames.append(smoothed_frame)
    
    smoothed_frames.append(smoothed_frame)

# Convert processed frames to a numpy array
processed_movie = np.array(processed_frames)

# Step 4: Save the processed movie as a new TIFF file (if desired)
##tiff.imwrite("processed_movie.tiff", processed_movie.astype(np.uint8))


# print filtered frames for comparison
for i in range(0, len(smoothed_frames), max(1, len(smoothed_frames) // 10)):  # Show 10 frames evenly spaced
    plt.figure()
    plt.imshow(smoothed_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
    
    
# binary frames
for i in range(0, len(binary_frames), max(1, len(binary_frames) // 10)):  # Show 10 frames evenly spaced
    plt.figure()
    plt.imshow(binary_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()

# Step 5: Check processed frames
for i in range(0, len(processed_frames), max(1, len(processed_frames) // 10)):  # Show 10 frames evenly spaced
    plt.figure()
    plt.imshow(processed_frames[i], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
    
    

import tifffile as tiff
import numpy as np
from skimage import img_as_ubyte

# Convert processed frames to 8-bit format for saving
processed_movie_8bit = [img_as_ubyte(frame) for frame in processed_frames]

# Specify the output file path
output_path = "processed_movie_with_display_settings.tif"

# Save the frames as a multi-page BigTIFF file (with display settings preserved)
tiff.imwrite(output_path, processed_movie_8bit, bigtiff=True)

print(f"Movie saved to: {output_path}")


'''


import tifffile as tiff
import numpy as np
import cv2
from skimage import exposure

# Step 1: Load the TIFF movie
tiff_movie = tiff.imread("M6688_Stack_G003_only.tif")

# Step 2: Define video properties
height, width = tiff_movie[0].shape  # Get dimensions of a single frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('original_visible_movie.mp4', fourcc, 15.0, (width, height))  # Adjust FPS as needed

# Step 3: Adjust contrast using histogram stretching
def adjust_contrast(frame):
    """Apply contrast stretching to enhance visibility."""
    # Rescale intensity to enhance contrast (0.5% to 99.5% percentile)
    p2, p98 = np.percentile(frame, (0.5, 99.5))
    adjusted_frame = exposure.rescale_intensity(frame, in_range=(p2, p98), out_range=(0, 255))
    return adjusted_frame.astype(np.uint8)

# Step 4: Write each frame to the video
for frame in tiff_movie:
##for i in range(0, len(tiff_movie), max(1, len(tiff_movie) // 10)): 
    ##frame = tiff_movie[i]
    # Adjust contrast
    adjusted_frame = adjust_contrast(frame)
    
    # Convert grayscale to BGR for OpenCV compatibility
    frame_bgr = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    
    # Write the frame to the video
    out.write(frame_bgr)

out.release()  # Finalize and save the video
print("Visible movie saved as 'original_visible_movie.mp4'")



############ --------------------------------------------%|











'''
# need to show full movies not just parts 
# make sure im stiching together 10 frames from input movie and 10 frames from output movie

import tifffile as tiff
import numpy as np
import cv2
from skimage import exposure
from skimage import img_as_ubyte

# Step 1: Load the TIFF movie
tiff_movie = tiff.imread("M6688_Stack_G003_only.tif")
processed_movie = np.array(processed_frames)  # Use the processed frames from your previous code

# Step 2: Define video properties
height, width = tiff_movie[0].shape  # Assuming both movies have the same resolution
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Alternative codec for compatibility
out = cv2.VideoWriter('stitched_movie.avi', fourcc, 15.0, (width * 2, height))  # Note: width is doubled

# Step 3: Adjust contrast using histogram stretching
def adjust_contrast(frame):
    """Apply contrast stretching to enhance visibility."""
    p2, p98 = np.percentile(frame, (0.5, 99.5))
    adjusted_frame = exposure.rescale_intensity(frame, in_range=(p2, p98), out_range=(0, 255))
    return adjusted_frame.astype(np.uint8)

# Step 4: Stitch and write each frame to the video
for original_frame, processed_frame in zip(tiff_movie, processed_movie):
    # Adjust contrast of the original frame
    original_adjusted = adjust_contrast(original_frame)
    
    # Ensure processed frame is in 8-bit format
    if processed_frame.max() <= 1:
        processed_frame = img_as_ubyte(processed_frame)
    
    # Convert both frames to BGR for compatibility with OpenCV
    original_bgr = cv2.cvtColor(original_adjusted, cv2.COLOR_GRAY2BGR)
    processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    
    # Concatenate frames horizontally
    stitched_frame = np.hstack((original_bgr, processed_bgr))
    
    # Write the stitched frame to the video
    out.write(stitched_frame)

out.release()  # Finalize and save the video
print("Stitched movie saved as 'stitched_movie.avi'")
'''
'''
import cv2
import numpy as np

# Load the two video files
video1_path = "original_visible_movie.mp4"
video2_path = "output_movie.mp4"
output_path = "stitched_movie.avi"

# Open video captures
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get properties from the first video
fps = int(cap1.get(cv2.CAP_PROP_FPS))
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video properties
output_width = frame_width * 2  # Side by side
output_height = frame_height
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Break if either video ends
    if not ret1 or not ret2:
        break

    # Resize frames if necessary (ensuring both frames have the same height)
    frame2 = cv2.resize(frame2, (frame_width, frame_height))

    # Concatenate frames horizontally
    stitched_frame = np.hstack((frame1, frame2))

    # Write the stitched frame to the output video
    out.write(stitched_frame)

# Release resources
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()

print(f"Stitched video saved as {output_path}")

# Saves the result as stitched_movie.avi
'''
'''
import cv2
import numpy as np
#import ffmpeg

# Input video files
video1_path = "original_visible_movie.mp4"
video2_path = "output_movie.mp4"
output_path = "stitched_movie.mp4"

# Open the first video
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get video properties from the first video
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

# Open the second video to ensure it has the same properties
frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps2 = int(cap2.get(cv2.CAP_PROP_FPS))

# Ensure both videos have the same dimensions and FPS
if (frame_width, frame_height, fps) != (frame_width2, frame_height2, fps2):
    print("Error: The videos have different resolutions or frame rates.")
    cap1.release()
    cap2.release()
    exit()

# Define the output video writer with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264-compatible codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to write frames from a video to output
def write_video_frames(cap, writer):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

# Write frames from both videos sequentially
write_video_frames(cap1, out)
write_video_frames(cap2, out)

# Release resources
cap1.release()
cap2.release()
out.release()

print(f"Successfully created {output_path}")
'''

import cv2

# Input video filenames
video1_path = "original_visible_movie.mp4"
video2_path = "output_movie.mp4"
output_path = "side_by_side.mp4"

# Open video files
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get video properties
frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ensure both videos have the same height
if frame_height1 != frame_height2:
    scale_factor = frame_height1 / frame_height2
    frame_width2 = int(frame_width2 * scale_factor)
    frame_height2 = frame_height1

# Output video properties
output_width = frame_width1 + frame_width2
output_height = frame_height1

# Define codec and create output video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break  # Stop when one video ends

    # Resize second frame if needed
    frame2 = cv2.resize(frame2, (frame_width2, frame_height2))

    # Concatenate frames horizontally
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write frame to output video
    out.write(combined_frame)

# Release resources
cap1.release()
cap2.release()
out.release()

print(f"Side-by-side video saved as: {output_path}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:58:05 2025

@author: sydney
"""
# CURRENTLY WORKING ON !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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


'''
Documentation
https://skeleton-analysis.org/stable/getting_started/getting_started.html#extracting-a-skeleton-from-an-image
'''





# script ---------------------------------------------------|

#file_path = '/Users/sydney/Desktop/Suel lab rotation/output_movie.mp4'
#video_frames = load_skeleton_movie(file_path)

height, width = None, None
frames = []
trifurcations = []
total = []
average = []
number = []

for i in range(178):# 178
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
    '''
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
    # we calculated the number of junctions by counting the number of branches that were junction to junction
    '''
    skeleton = csr.Skeleton(skeleton0)
    # Count junctions (vertices with degree >= 3)
    junction_counts = (skeleton.degrees >= 3).sum()
    trifurcations.append(junction_counts)
    
    # other branch data 
    
    
    
    # total branch length
    branch_lengths = skeleton.path_lengths()
    total_branch_length = branch_lengths.sum()
    total_branch_length = round(total_branch_length, 2)
    total.append(total_branch_length)
    
    # average branch length
    average_branch_length = branch_lengths.mean()
    average_branch_length = round(average_branch_length, 2)
    average.append(average_branch_length)
    
    # number of branches 
    num_branches = len(branch_lengths)
    number.append(num_branches)
    
    
print('the number of trifurcations is = ', trifurcations)
#  [17, 11, 9, 10, 8, 12, 10, 8, 11, 7, 6, 6, 7, 10, 6, 8, 4, 3, 4, 9, 6, 5, 6, 5, 5, 8, 5, 7, 6, 10, 8, 14, 6, 12, 9, 11, 9, 11, 9, 8, 9, 11, 7, 6, 5, 12, 9, 10, 17, 9, 18, 8, 11, 14, 11, 9, 10, 13, 9, 10, 10, 17, 14, 7, 13, 13, 13, 13, 8, 13, 12, 14, 8, 12, 12, 9, 13, 8, 10, 6, 8, 13, 6, 9, 12, 8, 9, 9, 9, 12, 13, 11, 11, 10, 12, 10, 12, 12, 11, 8, 12, 9, 7, 11, 10, 9, 8, 9, 11, 16, 13, 8, 15, 9, 8, 8, 9, 12, 12, 9, 9, 13, 11, 8, 13, 13, 13, 16, 12, 18, 11, 8, 11, 10, 12, 12, 12, 13, 6, 11, 12, 8, 12, 10, 9, 11, 10, 14, 16, 10, 12, 10, 9, 8, 7, 11, 12, 13, 9, 12, 7, 15, 14, 15, 8, 9, 7, 13, 6, 8, 14, 7, 7, 8, 7, 8, 12, 9]
print('the total branch lengeth is = ', total)
#  [12092.84, 10793.57, 10131.42, 10625.11, 10262.13, 10691.22, 10083.21, 9531.02, 11162.46, 8728.58, 8950.71, 9144.17, 9558.86, 9902.45, 8757.44, 9669.23, 8732.79, 8585.67, 8718.55, 10454.24, 9435.06, 9165.1, 9004.72, 8819.7, 8943.58, 10222.55, 8864.15, 9920.46, 9879.85, 10568.61, 10245.99, 11235.43, 9628.97, 10828.68, 11008.56, 11380.91, 10443.94, 11388.04, 11037.58, 11029.0, 10920.85, 11384.77, 10629.46, 9855.06, 9836.49, 11460.79, 11250.23, 11069.2, 12396.74, 10034.71, 12859.84, 10141.39, 10728.36, 12484.41, 10378.53, 10504.05, 10706.46, 11133.51, 10581.1, 12063.11, 10677.49, 12262.25, 11647.04, 10071.75, 12136.59, 11995.67, 10951.1, 11829.94, 11394.06, 12038.03, 10564.47, 11593.25, 10403.52, 11690.11, 11778.18, 11210.34, 11733.49, 10862.25, 10785.42, 10357.68, 11339.59, 11939.26, 10332.25, 11112.1, 12060.12, 10743.34, 10713.68, 10974.53, 10555.92, 11386.68, 11790.09, 12007.8, 11845.35, 11989.03, 11814.58, 11666.39, 11820.21, 12203.58, 11477.44, 9471.94, 12105.85, 11257.01, 9977.02, 10911.55, 11385.72, 10332.94, 10347.34, 11801.98, 10919.86, 12626.51, 12011.27, 10476.59, 12737.24, 11166.69, 10809.42, 10773.37, 10244.62, 11480.19, 11205.56, 10700.05, 10600.79, 12162.26, 11777.89, 9996.3, 12239.34, 11174.7, 12476.18, 12242.44, 11957.53, 13536.56, 10906.71, 10140.11, 11357.77, 10967.78, 11858.47, 11627.78, 11868.1, 11705.72, 10066.98, 11332.28, 11334.75, 11110.25, 11391.3, 11509.8, 11626.74, 11783.17, 12209.85, 12018.21, 13233.76, 12418.39, 12159.05, 12006.08, 11115.71, 11032.95, 11000.02, 12193.24, 12167.14, 12639.7, 12037.37, 11760.14, 11321.31, 13203.13, 12776.66, 12539.88, 11795.63, 12208.82, 10682.18, 12883.67, 11490.72, 11122.65, 13041.12, 11130.84, 10918.99, 11513.21, 11036.76, 12088.33, 11888.03, 12227.54]
print('the average branch length is = ', average)
# [366.45, 513.98, 562.86, 482.96, 733.01, 534.56, 530.7, 560.65, 558.12, 623.47, 688.52, 703.4, 637.26, 495.12, 729.79, 604.33, 1091.6, 1073.21, 1089.82, 696.95, 786.25, 916.51, 750.39, 881.97, 894.36, 681.5, 886.41, 826.7, 898.17, 660.54, 731.86, 468.14, 875.36, 541.43, 733.9, 599.0, 580.22, 569.4, 689.85, 735.27, 642.4, 599.2, 885.79, 895.91, 1229.56, 573.04, 703.14, 553.46, 442.74, 501.74, 367.42, 596.55, 510.87, 499.38, 471.75, 617.89, 535.32, 463.9, 529.05, 709.59, 533.87, 422.84, 447.96, 774.75, 505.69, 479.83, 456.3, 563.33, 876.47, 523.39, 440.19, 429.38, 693.57, 531.37, 535.37, 590.02, 488.9, 638.96, 539.27, 796.74, 596.82, 542.69, 939.3, 694.51, 574.29, 671.46, 669.6, 645.56, 620.94, 455.47, 561.43, 600.39, 592.27, 631.0, 537.03, 614.02, 422.15, 581.12, 573.87, 557.17, 550.27, 750.47, 712.64, 495.98, 669.75, 607.82, 689.82, 843.0, 545.99, 505.06, 500.47, 616.27, 489.89, 744.45, 720.63, 718.22, 569.15, 521.83, 533.6, 713.34, 623.58, 506.76, 535.36, 624.77, 556.33, 465.61, 567.1, 437.23, 543.52, 436.66, 574.04, 633.76, 597.78, 645.16, 592.92, 528.54, 565.15, 487.74, 915.18, 566.61, 539.75, 793.59, 517.79, 677.05, 775.12, 620.17, 763.12, 500.76, 472.63, 730.49, 607.95, 706.24, 694.73, 735.53, 846.16, 641.75, 579.39, 549.55, 708.08, 560.01, 808.67, 528.13, 532.36, 447.85, 786.38, 718.17, 667.64, 560.16, 957.56, 695.17, 543.38, 795.06, 727.93, 677.25, 735.78, 863.45, 516.87, 679.31]
print('the number of branches is = ', number)
# [33, 21, 18, 22, 14, 20, 19, 17, 20, 14, 13, 13, 15, 20, 12, 16, 8, 8, 8, 15, 12, 10, 12, 10, 10, 15, 10, 12, 11, 16, 14, 24, 11, 20, 15, 19, 18, 20, 16, 15, 17, 19, 12, 11, 8, 20, 16, 20, 28, 20, 35, 17, 21, 25, 22, 17, 20, 24, 20, 17, 20, 29, 26, 13, 24, 25, 24, 21, 13, 23, 24, 27, 15, 22, 22, 19, 24, 17, 20, 13, 19, 22, 11, 16, 21, 16, 16, 17, 17, 25, 21, 20, 20, 19, 22, 19, 28, 21, 20, 17, 22, 15, 14, 22, 17, 17, 15, 14, 20, 25, 24, 17, 26, 15, 15, 15, 18, 22, 21, 15, 17, 24, 22, 16, 22, 24, 22, 28, 22, 31, 19, 16, 19, 17, 20, 22, 21, 24, 11, 20, 21, 14, 22, 17, 15, 19, 16, 24, 28, 17, 20, 17, 16, 15, 13, 19, 21, 23, 17, 21, 14, 25, 24, 28, 15, 17, 16, 23, 12, 16, 24, 14, 15, 17, 15, 14, 23, 18]
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




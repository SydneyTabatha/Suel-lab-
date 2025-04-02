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

# Compute the Power Spectrum of the Skeletonized Image
def compute_power_spectrum(skeleton):
    # Compute the 2D FFT of the skeleton image
    fft_image = np.fft.fft2(skeleton)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift zero frequency to center
    power_spectrum = np.abs(fft_shifted) ** 2  # Compute power spectrum

    # Normalize and log-transform for better visualization
    power_spectrum_log = np.log1p(power_spectrum)  # log(1 + x) to avoid log(0)
    
    return power_spectrum_log

height, width = None, None
frames = []
trifurcations = []
total = []
average = []
number = []

total_power_spectrum = None
num_frames = 178 

for i in range(num_frames):# 178
    #image0 = iio.imread('M6688_Stack_G003_only.tif')[i]  # Example: using the first frame from the stack
    #image0 = iio.imread('M6800 BS3610 pH=6.tif')[i] 
    image0 = iio.imread('M6765_3610 YFP_chaotic pattern (1) (1).tif')[i] 
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
    
    
    # power spectrum analysis
    # Compute power spectrum of the skeletonized frame
    power_spectrum = compute_power_spectrum(skeleton0)
    #power_spectrum = compute_power_spectrum(image0)
    

    # Initialize or accumulate
    if total_power_spectrum is None:
        total_power_spectrum = power_spectrum
    else:
        total_power_spectrum += power_spectrum
    
    
print('the number of trifurcations is = ', trifurcations)
#  movie 1: [17, 11, 9, 10, 8, 12, 10, 8, 11, 7, 6, 6, 7, 10, 6, 8, 4, 3, 4, 9, 6, 5, 6, 5, 5, 8, 5, 7, 6, 10, 8, 14, 6, 12, 9, 11, 9, 11, 9, 8, 9, 11, 7, 6, 5, 12, 9, 10, 17, 9, 18, 8, 11, 14, 11, 9, 10, 13, 9, 10, 10, 17, 14, 7, 13, 13, 13, 13, 8, 13, 12, 14, 8, 12, 12, 9, 13, 8, 10, 6, 8, 13, 6, 9, 12, 8, 9, 9, 9, 12, 13, 11, 11, 10, 12, 10, 12, 12, 11, 8, 12, 9, 7, 11, 10, 9, 8, 9, 11, 16, 13, 8, 15, 9, 8, 8, 9, 12, 12, 9, 9, 13, 11, 8, 13, 13, 13, 16, 12, 18, 11, 8, 11, 10, 12, 12, 12, 13, 6, 11, 12, 8, 12, 10, 9, 11, 10, 14, 16, 10, 12, 10, 9, 8, 7, 11, 12, 13, 9, 12, 7, 15, 14, 15, 8, 9, 7, 13, 6, 8, 14, 7, 7, 8, 7, 8, 12, 9]
# movie 2: [56, 47, 45, 48, 42, 34, 40, 23, 23, 20, 8, 22, 11, 11, 10, 10, 14, 15, 14, 27, 15, 23, 12, 17, 15, 18, 25, 21, 17, 18, 17, 14, 13, 19, 23, 15, 19, 15, 17, 17, 15, 10, 17, 16, 13, 12, 10, 9, 8, 12, 19, 6, 9, 9, 21, 12, 15, 14, 12, 13, 9, 7, 9, 10, 10, 13, 9, 8, 8, 9, 9, 8, 9, 9, 12, 3, 5, 8, 9, 7, 9, 11, 10, 10, 9, 10, 8, 6, 8, 8, 11, 8, 9, 11, 11, 7, 10, 6, 11, 9, 6, 9, 11, 11, 9, 9, 8, 6, 6, 7, 8, 11, 6, 7, 10, 13, 10, 9, 9, 8, 8, 8, 12, 6, 4, 4, 8, 10, 7, 8, 10, 12, 8, 14, 16, 15, 12, 13, 8, 13, 13, 15, 14, 13, 13, 11, 12, 16, 8, 14, 13, 15, 16, 17, 11, 12, 18, 19, 17, 20, 22, 22, 18, 21, 15, 15, 16, 20, 14, 21, 9, 13, 18, 15, 10, 22, 14, 14]
# movie 3: [42, 42, 41, 41, 24, 32, 22, 14, 12, 10, 11, 8, 13, 12, 8, 9, 10, 9, 11, 7, 11, 10, 7, 8, 11, 8, 11, 10, 8, 8, 6, 2, 7, 3, 4, 8, 7, 3, 8, 9, 5, 5, 5, 7, 4, 9, 6, 11, 9, 5, 8, 8, 6, 6, 11, 9, 10, 15, 13, 19, 10, 11, 9, 8, 9, 10, 6, 10, 12, 9, 7, 11, 9, 8, 8, 12, 14, 11, 10, 11, 8, 14, 13, 13, 10, 13, 14, 9, 10, 8, 5, 7, 9, 12, 14, 8, 9, 9, 10, 10, 10, 10, 13, 9, 12, 9, 9, 12, 6, 11, 10, 15, 10, 8, 8, 9, 10, 10, 7, 7, 13, 10, 10, 8, 10, 12, 11, 14, 9, 10, 10, 8, 7, 9, 11, 10, 9, 11, 12, 10, 12, 7, 8, 12, 8, 13, 6, 10, 8, 11, 16, 11, 14, 11, 11, 10, 7, 7, 8, 11, 5, 7, 8, 15, 5, 7, 11, 11, 9, 7, 11, 6, 14, 11, 10, 11, 9, 10]

print('the total branch lengeth is = ', total)
#  movie 1: [12092.84, 10793.57, 10131.42, 10625.11, 10262.13, 10691.22, 10083.21, 9531.02, 11162.46, 8728.58, 8950.71, 9144.17, 9558.86, 9902.45, 8757.44, 9669.23, 8732.79, 8585.67, 8718.55, 10454.24, 9435.06, 9165.1, 9004.72, 8819.7, 8943.58, 10222.55, 8864.15, 9920.46, 9879.85, 10568.61, 10245.99, 11235.43, 9628.97, 10828.68, 11008.56, 11380.91, 10443.94, 11388.04, 11037.58, 11029.0, 10920.85, 11384.77, 10629.46, 9855.06, 9836.49, 11460.79, 11250.23, 11069.2, 12396.74, 10034.71, 12859.84, 10141.39, 10728.36, 12484.41, 10378.53, 10504.05, 10706.46, 11133.51, 10581.1, 12063.11, 10677.49, 12262.25, 11647.04, 10071.75, 12136.59, 11995.67, 10951.1, 11829.94, 11394.06, 12038.03, 10564.47, 11593.25, 10403.52, 11690.11, 11778.18, 11210.34, 11733.49, 10862.25, 10785.42, 10357.68, 11339.59, 11939.26, 10332.25, 11112.1, 12060.12, 10743.34, 10713.68, 10974.53, 10555.92, 11386.68, 11790.09, 12007.8, 11845.35, 11989.03, 11814.58, 11666.39, 11820.21, 12203.58, 11477.44, 9471.94, 12105.85, 11257.01, 9977.02, 10911.55, 11385.72, 10332.94, 10347.34, 11801.98, 10919.86, 12626.51, 12011.27, 10476.59, 12737.24, 11166.69, 10809.42, 10773.37, 10244.62, 11480.19, 11205.56, 10700.05, 10600.79, 12162.26, 11777.89, 9996.3, 12239.34, 11174.7, 12476.18, 12242.44, 11957.53, 13536.56, 10906.71, 10140.11, 11357.77, 10967.78, 11858.47, 11627.78, 11868.1, 11705.72, 10066.98, 11332.28, 11334.75, 11110.25, 11391.3, 11509.8, 11626.74, 11783.17, 12209.85, 12018.21, 13233.76, 12418.39, 12159.05, 12006.08, 11115.71, 11032.95, 11000.02, 12193.24, 12167.14, 12639.7, 12037.37, 11760.14, 11321.31, 13203.13, 12776.66, 12539.88, 11795.63, 12208.82, 10682.18, 12883.67, 11490.72, 11122.65, 13041.12, 11130.84, 10918.99, 11513.21, 11036.76, 12088.33, 11888.03, 12227.54]
# movie 2: [25420.37, 22469.43, 22715.88, 24115.43, 21613.67, 20731.13, 22996.27, 17965.95, 18861.07, 17543.06, 14936.7, 19176.86, 15836.35, 14955.1, 14631.71, 14643.06, 15449.68, 15540.54, 15822.33, 18181.39, 16469.63, 17371.94, 14683.56, 16806.67, 15305.99, 16930.3, 17549.32, 17047.63, 16293.67, 15908.96, 15598.78, 15962.21, 14375.76, 16833.94, 17909.27, 16568.13, 16729.58, 14676.31, 15544.5, 16152.54, 14840.74, 13771.9, 15820.3, 15348.45, 14595.13, 14271.26, 13892.4, 13508.36, 12755.37, 14134.17, 16273.23, 12995.85, 14120.07, 14098.64, 17016.27, 14424.26, 15298.5, 16152.72, 15007.82, 15786.02, 13805.85, 14060.27, 14355.06, 14852.89, 13932.8, 14484.7, 13497.04, 13861.32, 13233.81, 13686.68, 14770.57, 14456.05, 14136.81, 13650.03, 15119.98, 12450.72, 13825.4, 13532.15, 13791.76, 13804.78, 13984.77, 15107.46, 14462.14, 13583.39, 13532.02, 13686.47, 13863.98, 13445.43, 13995.21, 13164.27, 14699.05, 14584.8, 14326.18, 15234.55, 14289.91, 14264.48, 14713.63, 13638.26, 14499.22, 15500.8, 13229.56, 13712.82, 13973.23, 14227.94, 14368.88, 13769.16, 13925.4, 13349.19, 13462.84, 13397.34, 14073.85, 15315.25, 13469.4, 13908.57, 14403.14, 15845.01, 15218.71, 14420.24, 14227.98, 13522.55, 13747.29, 14301.18, 14446.21, 13066.81, 12425.9, 12273.19, 12809.99, 13216.46, 12727.42, 13869.92, 13843.28, 13822.98, 13253.83, 15321.06, 15346.87, 15684.24, 14904.03, 14516.86, 14535.34, 14382.2, 14567.72, 15523.85, 15282.95, 14855.94, 16112.65, 15729.14, 15511.37, 17403.94, 13887.52, 15956.75, 15529.5, 15978.14, 16492.8, 16645.88, 15096.31, 15875.6, 17355.36, 17663.53, 16705.64, 17817.46, 18166.36, 17954.01, 17155.16, 17884.98, 17092.64, 17210.82, 17189.91, 18394.84, 17290.94, 17665.68, 16495.21, 16321.66, 18573.01, 18341.68, 17045.22, 20214.33, 18848.68, 18695.87]
# movie 3:  [22799.69, 22061.08, 22308.12, 23245.67, 18646.14, 20145.85, 18176.0, 15994.28, 14561.84, 15202.01, 14888.59, 13626.17, 14735.7, 15528.93, 14632.88, 14844.12, 15131.88, 14382.17, 14671.86, 13014.67, 14841.1, 14632.84, 14532.07, 13700.8, 14965.12, 14095.22, 15027.8, 14113.53, 13529.4, 13072.12, 12588.82, 12071.65, 13959.98, 12488.94, 12148.6, 14591.57, 14107.36, 12892.84, 14328.87, 13799.61, 13019.33, 13183.6, 13545.8, 14381.57, 13437.88, 13300.48, 14367.97, 15699.54, 13971.58, 12831.23, 14564.61, 13988.51, 13634.29, 13638.83, 16024.82, 15511.15, 15585.84, 17239.15, 17546.7, 17804.66, 15196.76, 14745.18, 14861.18, 15029.96, 15605.23, 16313.49, 15264.54, 15819.82, 15923.65, 15129.6, 14532.89, 15774.24, 15026.72, 14411.48, 15163.38, 16851.71, 17101.45, 16145.68, 15032.39, 15689.18, 14553.73, 16263.1, 15656.06, 15709.01, 14910.46, 16455.95, 17014.61, 15188.49, 15475.97, 15183.94, 13785.17, 14660.8, 15272.21, 16544.62, 16420.34, 14638.06, 15227.68, 15145.9, 15173.65, 15266.69, 14427.68, 16092.06, 17326.86, 14819.39, 15936.38, 15260.25, 15741.64, 16282.43, 14595.94, 15469.83, 14661.36, 16950.31, 14915.23, 14344.26, 14584.24, 14679.93, 15165.19, 15295.24, 14483.13, 14447.81, 16363.19, 14679.14, 14744.68, 14520.0, 15170.89, 15251.27, 15300.85, 16194.42, 15617.94, 14888.33, 15302.34, 13968.95, 14128.36, 14653.33, 14710.04, 15197.17, 15010.41, 15480.02, 14955.65, 14951.93, 15125.2, 13897.91, 14711.24, 15338.53, 14897.14, 16027.3, 13489.57, 14400.5, 14304.76, 16039.82, 15703.84, 15648.08, 16289.23, 15593.4, 16578.85, 16194.07, 14156.69, 13931.03, 13575.05, 15492.82, 12642.94, 14237.66, 13869.69, 16091.01, 13010.11, 14438.34, 15641.47, 14772.44, 14118.97, 13746.5, 14551.26, 13260.34, 15197.35, 14778.38, 14526.88, 14519.07, 15060.84, 14587.97]

print('the average branch length is = ', average)
# movie 1: [366.45, 513.98, 562.86, 482.96, 733.01, 534.56, 530.7, 560.65, 558.12, 623.47, 688.52, 703.4, 637.26, 495.12, 729.79, 604.33, 1091.6, 1073.21, 1089.82, 696.95, 786.25, 916.51, 750.39, 881.97, 894.36, 681.5, 886.41, 826.7, 898.17, 660.54, 731.86, 468.14, 875.36, 541.43, 733.9, 599.0, 580.22, 569.4, 689.85, 735.27, 642.4, 599.2, 885.79, 895.91, 1229.56, 573.04, 703.14, 553.46, 442.74, 501.74, 367.42, 596.55, 510.87, 499.38, 471.75, 617.89, 535.32, 463.9, 529.05, 709.59, 533.87, 422.84, 447.96, 774.75, 505.69, 479.83, 456.3, 563.33, 876.47, 523.39, 440.19, 429.38, 693.57, 531.37, 535.37, 590.02, 488.9, 638.96, 539.27, 796.74, 596.82, 542.69, 939.3, 694.51, 574.29, 671.46, 669.6, 645.56, 620.94, 455.47, 561.43, 600.39, 592.27, 631.0, 537.03, 614.02, 422.15, 581.12, 573.87, 557.17, 550.27, 750.47, 712.64, 495.98, 669.75, 607.82, 689.82, 843.0, 545.99, 505.06, 500.47, 616.27, 489.89, 744.45, 720.63, 718.22, 569.15, 521.83, 533.6, 713.34, 623.58, 506.76, 535.36, 624.77, 556.33, 465.61, 567.1, 437.23, 543.52, 436.66, 574.04, 633.76, 597.78, 645.16, 592.92, 528.54, 565.15, 487.74, 915.18, 566.61, 539.75, 793.59, 517.79, 677.05, 775.12, 620.17, 763.12, 500.76, 472.63, 730.49, 607.95, 706.24, 694.73, 735.53, 846.16, 641.75, 579.39, 549.55, 708.08, 560.01, 808.67, 528.13, 532.36, 447.85, 786.38, 718.17, 667.64, 560.16, 957.56, 695.17, 543.38, 795.06, 727.93, 677.25, 735.78, 863.45, 516.87, 679.31]
# movie 2: [229.01, 226.96, 241.66, 253.85, 257.31, 283.99, 291.09, 345.5, 471.53, 548.22, 933.54, 532.69, 879.8, 787.11, 770.09, 665.59, 572.21, 597.71, 608.55, 378.78, 633.45, 423.71, 638.42, 542.15, 546.64, 497.95, 398.85, 460.75, 581.92, 513.19, 503.19, 613.93, 598.99, 467.61, 459.21, 637.24, 539.66, 543.57, 536.02, 556.98, 549.66, 810.11, 510.33, 529.26, 561.35, 648.69, 817.2, 844.27, 850.36, 642.46, 508.54, 1082.99, 882.5, 829.33, 459.9, 627.14, 566.61, 702.29, 714.66, 751.72, 766.99, 1278.21, 957.0, 825.16, 733.31, 603.53, 843.57, 990.09, 945.27, 912.45, 923.16, 1204.67, 831.58, 853.13, 720.0, 2490.14, 1728.17, 845.76, 861.98, 1254.98, 874.05, 795.13, 850.71, 754.63, 845.75, 805.09, 1066.46, 1222.31, 999.66, 940.31, 773.63, 1121.91, 1023.3, 846.36, 714.5, 1188.71, 919.6, 1239.84, 763.12, 1107.2, 1102.46, 914.19, 698.66, 790.44, 957.93, 809.95, 1071.18, 1112.43, 1223.89, 1116.44, 1082.6, 806.07, 1224.49, 1264.42, 900.2, 754.52, 895.22, 961.35, 948.53, 965.9, 981.95, 1100.09, 628.1, 1306.68, 1775.13, 1753.31, 985.38, 777.44, 979.03, 1066.92, 814.31, 628.32, 883.59, 638.38, 590.26, 627.37, 745.2, 604.87, 1118.1, 719.11, 693.7, 646.83, 694.68, 707.43, 732.39, 925.24, 738.64, 644.59, 1068.27, 725.31, 776.48, 694.7, 659.71, 554.86, 838.68, 755.98, 578.51, 588.78, 618.73, 574.76, 519.04, 512.97, 571.84, 526.03, 683.71, 748.3, 636.66, 574.84, 751.78, 477.45, 1099.68, 741.89, 619.1, 797.46, 1065.33, 594.54, 724.95, 812.86]
# movie 3: [253.33, 247.88, 262.45, 276.73, 327.13, 300.68, 349.54, 533.14, 502.13, 844.56, 744.43, 757.01, 613.99, 647.04, 975.53, 989.61, 890.11, 898.89, 698.66, 813.42, 742.05, 731.64, 1211.01, 856.3, 748.26, 1084.25, 715.61, 882.1, 1040.72, 933.72, 1144.44, 2011.94, 1269.09, 2081.49, 1518.57, 972.77, 1085.18, 2148.81, 1023.49, 766.65, 1084.94, 1464.84, 1505.09, 1106.27, 1679.74, 738.92, 1306.18, 713.62, 821.86, 1425.69, 1040.33, 874.28, 1048.79, 974.2, 728.4, 969.45, 974.12, 718.3, 797.58, 613.95, 949.8, 702.15, 928.82, 1073.57, 975.33, 906.31, 1387.69, 832.62, 796.18, 1080.69, 1321.17, 830.22, 939.17, 900.72, 1010.89, 886.93, 712.56, 896.98, 939.52, 784.46, 1119.52, 625.5, 680.7, 714.05, 784.76, 658.24, 708.94, 893.44, 859.78, 1168.0, 1531.69, 1221.73, 954.51, 787.84, 713.93, 1126.0, 1087.69, 841.44, 842.98, 954.17, 759.35, 946.59, 825.09, 926.21, 796.82, 1017.35, 983.85, 740.11, 1122.76, 703.17, 771.65, 651.93, 877.37, 1024.59, 1121.86, 917.5, 947.82, 805.01, 1316.65, 1203.98, 743.78, 917.45, 921.54, 1116.92, 948.18, 762.56, 765.04, 704.11, 1041.2, 992.56, 900.14, 1164.08, 1009.17, 976.89, 865.3, 844.29, 882.97, 774.0, 679.8, 934.5, 720.25, 1158.16, 1131.63, 730.41, 993.14, 728.51, 1124.13, 757.92, 1021.77, 844.2, 541.51, 680.35, 626.51, 779.67, 789.47, 899.67, 1088.98, 1160.92, 1044.23, 911.34, 1264.29, 1095.2, 815.86, 618.89, 1301.01, 1203.2, 868.97, 738.62, 882.44, 981.89, 692.92, 1105.03, 607.89, 777.81, 807.05, 764.16, 941.3, 810.44]

print('the number of branches is = ', number)
# movie 1: [33, 21, 18, 22, 14, 20, 19, 17, 20, 14, 13, 13, 15, 20, 12, 16, 8, 8, 8, 15, 12, 10, 12, 10, 10, 15, 10, 12, 11, 16, 14, 24, 11, 20, 15, 19, 18, 20, 16, 15, 17, 19, 12, 11, 8, 20, 16, 20, 28, 20, 35, 17, 21, 25, 22, 17, 20, 24, 20, 17, 20, 29, 26, 13, 24, 25, 24, 21, 13, 23, 24, 27, 15, 22, 22, 19, 24, 17, 20, 13, 19, 22, 11, 16, 21, 16, 16, 17, 17, 25, 21, 20, 20, 19, 22, 19, 28, 21, 20, 17, 22, 15, 14, 22, 17, 17, 15, 14, 20, 25, 24, 17, 26, 15, 15, 15, 18, 22, 21, 15, 17, 24, 22, 16, 22, 24, 22, 28, 22, 31, 19, 16, 19, 17, 20, 22, 21, 24, 11, 20, 21, 14, 22, 17, 15, 19, 16, 24, 28, 17, 20, 17, 16, 15, 13, 19, 21, 23, 17, 21, 14, 25, 24, 28, 15, 17, 16, 23, 12, 16, 24, 14, 15, 17, 15, 14, 23, 18]
# movie 2: [111, 99, 94, 95, 84, 73, 79, 52, 40, 32, 16, 36, 18, 19, 19, 22, 27, 26, 26, 48, 26, 41, 23, 31, 28, 34, 44, 37, 28, 31, 31, 26, 24, 36, 39, 26, 31, 27, 29, 29, 27, 17, 31, 29, 26, 22, 17, 16, 15, 22, 32, 12, 16, 17, 37, 23, 27, 23, 21, 21, 18, 11, 15, 18, 19, 24, 16, 14, 14, 15, 16, 12, 17, 16, 21, 5, 8, 16, 16, 11, 16, 19, 17, 18, 16, 17, 13, 11, 14, 14, 19, 13, 14, 18, 20, 12, 16, 11, 19, 14, 12, 15, 20, 18, 15, 17, 13, 12, 11, 12, 13, 19, 11, 11, 16, 21, 17, 15, 15, 14, 14, 13, 23, 10, 7, 7, 13, 17, 13, 13, 17, 22, 15, 24, 26, 25, 20, 24, 13, 20, 21, 24, 22, 21, 22, 17, 21, 27, 13, 22, 20, 23, 25, 30, 18, 21, 30, 30, 27, 31, 35, 35, 30, 34, 25, 23, 27, 32, 23, 37, 15, 22, 30, 23, 16, 34, 26, 23]
# movie 3: [90, 89, 85, 84, 57, 67, 52, 30, 29, 18, 20, 18, 24, 24, 15, 15, 17, 16, 21, 16, 20, 20, 12, 16, 20, 13, 21, 16, 13, 14, 11, 6, 11, 6, 8, 15, 13, 6, 14, 18, 12, 9, 9, 13, 8, 18, 11, 22, 17, 9, 14, 16, 13, 14, 22, 16, 16, 24, 22, 29, 16, 21, 16, 14, 16, 18, 11, 19, 20, 14, 11, 19, 16, 16, 15, 19, 24, 18, 16, 20, 13, 26, 23, 22, 19, 25, 24, 17, 18, 13, 9, 12, 16, 21, 23, 13, 14, 18, 18, 16, 19, 17, 21, 16, 20, 15, 16, 22, 13, 22, 19, 26, 17, 14, 13, 16, 16, 19, 11, 12, 22, 16, 16, 13, 16, 20, 20, 23, 15, 15, 17, 12, 14, 15, 17, 18, 17, 20, 22, 16, 21, 12, 13, 21, 15, 22, 12, 19, 14, 19, 29, 23, 26, 20, 21, 18, 13, 12, 13, 17, 10, 13, 17, 26, 10, 12, 18, 20, 16, 14, 21, 12, 25, 19, 18, 19, 16, 18]

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

# unsure of branch length as a metric - average branch length per frame?

# Compute the average power spectrum
avg_power_spectrum = total_power_spectrum / num_frames

# Plot the average power spectrum
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(avg_power_spectrum, cmap='inferno')
ax.set_title("Average Power Spectrum Across Frames")
ax.axis('off')

plt.show()


# this gives me the spatial frequency representation averaged over time
# last question - do i apply spectral analysis to the raw data or the skeleton?














# to do next

#combine branch types 2 and 1 to get accurate number of junctions





# plug this time sereies of junctions into chaos decision tree algorithm
# read about entropy gurol suggestion

# see if its accurate enough both the skeleton and the number of nodes
# then try with different movies
# then try with different 1-d metrics


# first movie
# [12, 8, 7, 8, 5, 10, 8, 7, 9, 6, 5, 5, 5, 7, 4, 5, 3, 1, 3, 8, 4, 4, 5, 4, 4, 6, 4, 6, 5, 8, 7, 11, 4, 11, 8, 8, 7, 9, 6, 6, 7, 10, 6, 4, 4, 11, 8, 9, 15, 7, 14, 6, 9, 12, 8, 8, 7, 11, 7, 9, 8, 15, 12, 6, 12, 11, 10, 12, 6, 10, 9, 11, 7, 9, 9, 4, 11, 5, 7, 5, 5, 12, 4, 7, 10, 6, 7, 7, 7, 10, 12, 10, 10, 7, 9, 7, 7, 10, 10, 6, 10, 8, 5, 8, 9, 8, 7, 8, 9, 15, 11, 7, 13, 8, 6, 7, 7, 10, 9, 8, 7, 11, 9, 6, 11, 10, 12, 15, 11, 16, 10, 6, 10, 9, 11, 10, 10, 11, 5, 8, 10, 7, 10, 9, 8, 9, 8, 12, 12, 8, 11, 8, 7, 7, 5, 10, 11, 10, 7, 10, 5, 12, 11, 13, 6, 7, 5, 11, 4, 6, 13, 6, 5, 5, 5, 7, 11, 8]
# stochastic from decision tree algorithm

# num of endpoints 

#[10, 7, 7, 7, 3, 4, 6, 8, 7, 7, 4, 6, 6, 9, 6, 7, 3, 5, 2, 3, 6, 5, 4, 5, 3, 6, 3, 3, 4, 2, 4, 6, 3, 4, 3, 4, 7, 5, 4, 6, 7, 3, 3, 4, 1, 4, 5, 6, 5, 7, 13, 9, 5, 7, 9, 7, 9, 8, 5, 4, 7, 7, 9, 5, 5, 9, 9, 3, 2, 7, 10, 10, 6, 7, 8, 8, 9, 6, 8, 6, 8, 5, 4, 5, 6, 7, 4, 7, 7, 10, 3, 7, 7, 7, 6, 6, 13, 6, 5, 9, 6, 3, 7, 11, 4, 7, 6, 1, 6, 2, 7, 6, 7, 3, 6, 4, 7, 8, 6, 3, 7, 7, 7, 6, 5, 7, 5, 8, 8, 8, 3, 8, 5, 4, 4, 7, 4, 8, 4, 7, 6, 4, 6, 4, 3, 4, 2, 5, 8, 2, 4, 4, 5, 4, 5, 5, 6, 7, 6, 6, 4, 5, 6, 8, 6, 7, 8, 7, 6, 6, 6, 7, 9, 10, 8, 4, 6, 9]




# 2nd movie
#[6, 21, 9, 10, 9, 6, 11, 12, 12, 22, 14, 20, 11, 15, 11, 17, 22, 20, 16, 16, 16, 13, 11, 15, 22, 12, 16, 11, 15, 16, 12, 9, 15, 14, 11, 10, 9, 7, 6, 9, 16, 3, 8, 8, 18, 10, 13, 12, 11, 11, 6, 6, 8, 8, 8, 11, 7, 7, 7, 8, 8, 7, 7, 8, 11, 2, 4, 6, 8, 6, 8, 10, 9, 8, 7, 9, 7, 5, 7, 7, 9, 7, 8, 10, 10, 6, 9, 5, 10, 8, 5, 8, 9, 9, 7, 7, 7, 4, 5, 6, 7, 9, 5, 6, 9, 11, 8, 7, 7, 6, 7, 7, 11, 5, 3, 3, 7, 9, 5, 7, 8, 11, 6, 12, 15, 14, 10, 11, 6, 11, 11, 13, 12, 12, 10, 10, 10, 15, 7, 12, 12, 14, 15, 15, 9, 9, 17, 16, 15, 19, 21, 21, 16, 19, 14, 13, 15, 18, 13, 18, 8, 12, 15, 14, 8, 21, 12, 13]
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




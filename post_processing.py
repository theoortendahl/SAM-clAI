import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import os
from skimage.measure import regionprops
from skimage.filters import threshold_yen

# Load image data
labelled = tifffile.imread('label_path').astype(np.int16)
greyscale = tifffile.imread('image_path')

# Get greyscale threshold
threshold = threshold_yen(greyscale)

# Get region properties
props = regionprops(labelled, intensity_image=greyscale)

for prop in props:
    label = prop.label
    average = prop.mean_intensity
    

    print(f'Particle {label}: average={average:.2f}')
    
    if average < threshold:
        print(f'Removed object: {label}')
        labelled[labelled == label] = 0


tifffile.imwrite('label_path_out', labelled)

# SAM-clAI
This is a repository for training and using the Segment Anything Model 2 for 3D images of granular materials.

## Instructions for running SAM 2

1. The Segment Anything Model 2 is a prerequisite for this repository. Please follow the instructions here: https://github.com/facebookresearch/sam2/blob/main/README.md

2. The necessary code for running SAM in this repository is found in run_sam.py. This file should be placed in "sam2/". ( not "sam2/sam2/").

3. One folder containing each frame of the 3D image (in jpg-format) and a second folder containing input points (csv-file with 'X' and 'Y' column for each slice) should be placed in "sam2/notebooks/".

4. cd to the first "sam2/" folder and run "python run_sam.py"

## Instructions for training SAM 2

For training, the TRAIN.py folder should be placed in "sam2/training/".

## Acknowledgments

This repository is heavily dependent on the main Segment Anything Model 2 repository: https://github.com/facebookresearch/sam2

TRAIN.py is adapted from the following github repository: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code


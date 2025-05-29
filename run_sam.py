import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import tifffile
import skimage.measure as skim
import math
import cv2
from sys import exit

if torch.cuda.is_available():
    print('ja')
    device = torch.device("cuda")
    
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor


#Trained checkpoint, mean_iou=0.71
sam2_checkpoint = "./checkpoints/sam2_checkpoint_40000.pt"

#Checkpoint trained on clay images
#sam2_checkpoint = "./checkpoints/sam2_train_checkpoint.pt"


model_cfg = "./configs/sam2/sam2_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap='grey')


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "notebooks/crop1_600_jpgs"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

##### Manual inputs

# Image dimensions
nr_of_slices = len(os.listdir(video_dir)) #len(frame_names)
print('Nr of slices are: ' + str(nr_of_slices), flush=True)
img_size_x = 600
img_size_y = 600

# Threshold
overlap_threshold = 100
area_threshold = 6000

inference_state = predictor.init_state(video_path=video_dir)

# Maxima points directory
max_directory = "notebooks/maximas_crop1_600"

def get_XY(file, get='X'):
    filepath = os.path.join(max_directory, os.fsdecode(file))
    df_file = pd.read_csv(filepath, delimiter=',')
    return df_file[get]

def addPoint(x, y, point_id, show=True):

    points = np.array([[x[point_id], y[point_id]]], dtype=np.float32)     # Particles on first slice:  132, 214  353, 150   282, 468   125, 355   [31, 111]  40, 377
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=point_id,
        points=points,
        labels=labels,
    )
    if show:
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
       	show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        #current_dir = os.getcwd()
        #output_dir = os.path.dirname(os.path.dirname(current_dir))
        path = os.path.join('output_masks', 'frame' + str(ann_frame_idx) + str(point_id) + '.jpg')
        plt.savefig(path)
        plt.show()
    else:
        None

def segment_particle(prev_masks):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    i = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[local_frame_idx] > 0.0).cpu().numpy()
            for local_frame_idx, out_obj_id in enumerate(out_obj_ids)
        }
        image = np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        mask = video_segments[out_frame_idx][out_obj_ids[0]][0]
        #if out_frame_idx == end_frame:
            #break
	
        intersect = (mask != 0) * (prev_masks[out_frame_idx, :, :] != 0)
        if intersect.sum() > overlap_threshold:
            print('Particle overlap is too large.')
            return None
        else:
            None

	
        if i != 0:
            area = np.sum(mask)
            eq_radius = np.sqrt((area/np.pi))
            center1 = skim.centroid(mask, spacing=None)
            diff = np.sqrt(np.sum(np.square(center1 - center)))
            area_diff = np.abs(area - np.sum(video_segments[out_frame_idx - 1][out_obj_ids[0]]))
            if area == 0:
                print('Area is 0')
                return video_segments
            else:
                area_ratio = area_diff / area

            if area > area_threshold:
                print('Particle is too large. Area is:')
                print(str(area))
                return None
               
            else:
                None

            
            #print('Movement of mask: ' + str(diff) + ' pixels')
            #print('Diff in volume of mask: ' + str(area_diff) + ' pixels')
            #print('Masks area is: ' + str(area) + ' pixels. Its eq. radius is: ' + str(eq_radius))

            if diff > eq_radius or area_ratio > 0.5:
                print('Movement or change in area is too large')
                return video_segments
            elif diff == 0 and area_diff == 0:
                print('no diff')
                return video_segments
            elif math.isnan(diff) or math.isnan(area_diff):
                print('isnan')
                return video_segments
            else:
                None
        else:

            i += 1
        
	
        center = skim.centroid(mask, spacing=None)
    return video_segments

def render_results(video_segments):
    # render the segmentation results every few frames
    vis_frame_stride = 10
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

def save(img, maskNr, sliceNr):

    # From numpy array to PIL image
    print('Saving particle at:')

    path = os.path.join('output_masks', 'mask' + str(maskNr) + 'slice' + str(sliceNr) + '.jpg')
    print(str(path))
    img = np.reshape(img, shape=(1800, 1800))
    image = Image.fromarray(img)
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.savefig(path)
    plt.show()
    """
    if maskNr > 100:
        print('Mask Nr is higher than 100!')
        path_all = os.path.join('output_masks', 'all_masks.tif')
        tifffile.imwrite(path_all, out_mask_all_particles)
        exit(0)
    elif sliceNr > 2:
        print('Slice Nr is higher than 2!')
        path_all = os.path.join('output_masks', 'all_masks.tif')
        tifffile.imwrite(path_all, out_mask_all_particles)
        exit(0)
    else:
        None
    """
    # Save as JPEG
    #image.save(path, 'JPEG')

show_point = False


files = os.listdir(max_directory)

out_mask_all_particles = np.zeros((nr_of_slices, img_size_y, img_size_x))


for ann_frame_idx, file in enumerate(files):
    filepath = os.path.join(max_directory, os.fsdecode(file))
    df_f = pd.read_csv(filepath, delimiter=',')

    x = get_XY(file, get='X')
    y = get_XY(file, get='Y')

    
    counting_prev = 0
    for ann_obj_id in range(len(x)):

        predictor.reset_state(inference_state)

        if out_mask_all_particles[ann_frame_idx][y[ann_obj_id]][x[ann_obj_id]] != 0:
            print('Particle:' + str(ann_obj_id) + ' present on previous.')
            continue
        else:
            #double = False
            print(ann_obj_id)
            addPoint(x, y, ann_obj_id, show=show_point)
            video_segments = segment_particle(out_mask_all_particles)

        if video_segments == None:
            continue
        else:
            None


        i = 0
        k = ann_frame_idx
        out_mask = np.zeros((nr_of_slices, img_size_y, img_size_x))
        while i == 0:
            try:
                out_mask[k, :, :] = video_segments[k][ann_obj_id][0]
                k += 1
            except:
                if k == nr_of_slices:
                    i = 1
                else:
                    out_mask[k, :, :] = np.zeros((img_size_y, img_size_x))
                    i = 1


        #plt.imshow(out_mask[0, :, :])
        #plt.show()

        if out_mask.dtype != np.uint8:
            out_mask = ((out_mask - out_mask.min()) / (out_mask.max() - out_mask.min()) * 255).astype(np.uint8)

        out_mask_all_particles += (out_mask * (1/255)) * (ann_obj_id+1+counting_prev)
        #out_masks_frame += out_mask

        print('Mask' + str(ann_obj_id) + ' at frame' + str(ann_frame_idx) + ' is done.')
        #save(out_mask[ann_frame_idx, :,:], ann_obj_id, ann_frame_idx)
        #tifffile.imwrite(os.path.join(r'C:\Users\stick\Desktop\sam2res', 'out' + str(ann_obj_id) + 'in' + str(ann_frame_idx) + 'mask.tif'), (out_mask * (1/255)) * (ann_obj_id+1+counting_prev))

        #print(str(ann_obj_id+1) + ' Particles done on frame: ' + str(ann_frame_idx))

        if ann_obj_id == (len(x) - 1):
            counting_prev += ann_obj_id + 1

        
path_all = os.path.join('output_masks', 'all_masks.tif')
tifffile.imwrite(path_all, out_mask_all_particles)
 
        
            

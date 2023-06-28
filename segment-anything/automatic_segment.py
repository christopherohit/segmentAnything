import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from frr import FastReflectionRemoval

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse
import os

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_anns(anns, threshold=(100, 400), image=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        area_obj = ann['area']
        if (area_obj >= threshold[0]) and (area_obj <= threshold[1]):  
            color_mask = np.concatenate([(1,0,0), [0.35]])     
            img[m] = color_mask

            image = cv2.circle(image, np.asarray(ann['point_coords'][0], dtype=np.int64), 2, (0,0,255), 2)
    ax.imshow(img)
    return image


def load_model(model_path="weights/sam_vit_l_0b3195.pth", model_type = "vit_l", device = "cuda"):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else: device = "cpu"

    sam_checkpoint = model_path
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def preprocessing(img):
    # instantiate the algoroithm class
    alg = FastReflectionRemoval(h = 0.2)
    # run the algorithm and get result of shape (H, W, C)
    norm_img = img / 255
    dereflected_img = alg.remove_reflection(norm_img)

    img = np.asarray(dereflected_img * 255, dtype=np.int32)
    img = cv2.convertScaleAbs(img)

    blur_img = cv2.GaussianBlur(img, (3,3),0)

    return blur_img


def get_mask(image, sam, points_per_side=256, \
            pred_iou_thresh=0.88, \
            stability_score_thresh=0.5, \
            crop_n_layers=1,\
            crop_nms_thresh = 0.75 , \
            crop_overlap_ratio = 512/1500, \
            crop_n_points_downscale_factor=4,\
            min_mask_region_area=50,):
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(image)
    return masks



def main():
    folder_img = "data-test"
    output_path = "output"
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # for img in tqdm(os.listdir(folder_img)):
    #     input_path = os.path.join(folder_img, img)
    #     sam = load_model()
    #     image = cv2.imread(input_path)
    #     processed_image = preprocessing(image)

    #     masks = get_mask(
    #         image=processed_image,
    #         sam=sam)

    #     plt.figure(figsize=(50,50))
    #     plt.imshow(image)
    #     plot_image = show_anns(anns=masks, image=image)
    #     plt.axis('off')

    #     base_name = os.path.basename(input_path).split('.')[0]
    #     cv2.putText(plot_image, str(len(masks)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #     cv2.imwrite(os.path.join(output_path, f'{base_name}_outputpoint.jpg'), plot_image)
    #     plt.savefig(os.path.join(output_path, f'{base_name}_outputmask.jpg'))
    sam = load_model()
    image = cv2.imread('data-test/thumbnail_IMG_3810.jpg')
    processed_image = preprocessing(image)

    masks = get_mask(
        image=processed_image,
        sam=sam)

    plt.figure(figsize=(50,50))
    plt.imshow(image)
    plot_image = show_anns(anns=masks, image=image)
    plt.axis('off')

    base_name = os.path.basename('data-test/thumbnail_IMG_3808.jpg').split('.')[0]
    cv2.putText(plot_image, str(len(masks)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite(os.path.join(output_path, f'{base_name}_outputpoint.jpg'), plot_image)
    plt.savefig(os.path.join(output_path, f'{base_name}_outputmask.jpg'))


if __name__=='__main__':
    main()


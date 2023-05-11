import cv2
import numpy as np
import os
from pycocotools import mask as coco_mask
from pytoshop import layers
import pytoshop
from pytoshop.enums import BlendMode
from datetime import datetime

from scripts.layer_divider_modules.installation import *


def generate_random_color():
    return np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)


def create_base_layer(image):
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return [rgba_image]


def create_mask_layers(image, masks):
    layer_list = []

    for result in masks:
        rle = result['segmentation']
        mask = coco_mask.decode(rle).astype(np.uint8)
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba_image[..., 3] = cv2.bitwise_and(rgba_image[..., 3], rgba_image[..., 3], mask=mask)

        layer_list.append(rgba_image)

    return layer_list


def create_mask_gallery(image, masks):
    mask_array_list = []
    label_list = []

    for index, result in enumerate(masks):
        rle = result['segmentation']
        mask = coco_mask.decode(rle).astype(np.uint8)

        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba_image[..., 3] = cv2.bitwise_and(rgba_image[..., 3], rgba_image[..., 3], mask=mask)

        mask_array_list.append(rgba_image)
        label_list.append(f'Part {index}')

    return [[img, label] for img, label in zip(mask_array_list, label_list)]


def create_mask_combined_images(image, masks):
    final_result = np.zeros_like(image)

    for result in masks:
        rle = result['segmentation']
        mask = coco_mask.decode(rle).astype(np.uint8)

        color = generate_random_color()
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color

        final_result = cv2.addWeighted(final_result, 1, colored_mask, 0.5, 0)

    combined_image = cv2.addWeighted(image, 1, final_result, 0.5, 0)
    return [combined_image, "masked"]


def insert_psd_layer(psd, image_data, layer_name, blending_mode):
    channel_data = [layers.ChannelImageData(image=image_data[:, :, i], compression=1) for i in range(4)]

    layer_record = layers.LayerRecord(
        channels={-1: channel_data[3], 0: channel_data[0], 1: channel_data[1], 2: channel_data[2]},
        top=0, bottom=image_data.shape[0], left=0, right=image_data.shape[1],
        blend_mode=blending_mode,
        name=layer_name,
        opacity=255,
    )
    psd.layer_and_mask_info.layer_info.layer_records.append(layer_record)
    return psd


def save_psd(input_image_data, layer_data, layer_names, blending_modes, output_path):
    psd_file = pytoshop.core.PsdFile(num_channels=3, height=input_image_data.shape[0], width=input_image_data.shape[1])
    psd_file.layer_and_mask_info.layer_info.layer_records.clear()

    for index, layer in enumerate(layer_data):
        psd_file = insert_psd_layer(psd_file, layer, layer_names[index], blending_modes[index])

    with open(output_path, 'wb') as output_file:
        psd_file.write(output_file)


def save_psd_with_masks(image, masks, output_path):
    original_layer = create_base_layer(image)
    mask_layers = create_mask_layers(image, masks)
    names = [f'Part {i}' for i in range(len(mask_layers))]
    modes = [BlendMode.normal] * (len(mask_layers)+1)
    save_psd(image, original_layer+mask_layers, ['Original_Image']+names, modes, output_path)


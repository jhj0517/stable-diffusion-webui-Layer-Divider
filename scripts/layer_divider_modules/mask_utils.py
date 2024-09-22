import cv2
import numpy as np
from typing import Dict, List, Tuple
import colorsys
from pytoshop import layers
from pytoshop.enums import BlendMode
from pytoshop.core import PsdFile

from scripts.layer_divider_modules.constants import DEFAULT_COLOR, DEFAULT_PIXEL_SIZE


def decode_to_mask(seg: np.ndarray[np.bool_] | np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """Decode to uint8 mask from bool to deal with as images"""
    if isinstance(seg, np.ndarray) and seg.dtype == np.bool_:
        return seg.astype(np.uint8) * 255
    else:
        return seg.astype(np.uint8)


def invert_masks(masks: List[Dict]) -> List[Dict]:
    """Invert the masks. Used for background masking"""
    inverted = 1 - masks
    return inverted


def generate_random_color() -> Tuple[int, int, int]:
    """Generate random color in RGB format"""
    h = np.random.randint(0, 360)
    s = np.random.randint(70, 100) / 100
    v = np.random.randint(70, 100) / 100
    r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def create_base_layer(image: np.ndarray) -> List[np.ndarray]:
    """Create a base layer from the image. Used to keep original image"""
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return [rgba_image]


def create_mask_layers(
    image: np.ndarray,
    masks: List[Dict]
) -> List[np.ndarray]:
    """
    Create list of images with mask data. Masks are sorted by area in descending order.

    Args:
        image: Original image
        masks: List of mask data

    Returns:
        List of RGBA images
    """
    layer_list = []
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    for info in sorted_masks:
        rle = info['segmentation']
        mask = decode_to_mask(rle)

        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba_image[..., 3] = cv2.bitwise_and(rgba_image[..., 3], rgba_image[..., 3], mask=mask)

        layer_list.append(rgba_image)

    return layer_list


def create_mask_gallery(
    image: np.ndarray,
    masks: List[Dict]
) -> List:
    """
    Create list of images with mask data. Masks are sorted by area in descending order. Specially used for gradio
    Gallery component. each element has image and label, where label is the part number.

    Args:
        image: Original image
        masks: List of mask data

    Returns:
        List of [image, label] pairs
    """
    mask_array_list = []
    label_list = []

    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    for index, info in enumerate(sorted_masks):
        rle = info['segmentation']
        mask = decode_to_mask(rle)

        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba_image[..., 3] = cv2.bitwise_and(rgba_image[..., 3], rgba_image[..., 3], mask=mask)

        mask_array_list.append(rgba_image)
        label_list.append(f'Part {index}')

    return [[img, label] for img, label in zip(mask_array_list, label_list)]


def create_mask_combined_images(
    image: np.ndarray,
    masks: List[Dict]
) -> List:
    """
    Create an image with colored masks. Each mask is colored with a random color and blended with the original image.

    Args:
        image: Original image
        masks: List of mask data

    Returns:
        [image, label] pairs
    """
    final_result = np.zeros_like(image)
    used_colors = set()

    for info in masks:
        rle = info['segmentation']
        mask = decode_to_mask(rle)

        while True:
            color = generate_random_color()
            if color not in used_colors:
                used_colors.add(color)
                break

        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        blended = cv2.addWeighted(image, 0.3, colored_mask, 0.7, 0)
        final_result = np.where(mask[:, :, np.newaxis] > 0, blended, final_result)

    combined_image = np.where(final_result != 0, final_result, image)

    hsv = cv2.cvtColor(combined_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return [enhanced, "Masked"]


def create_mask_pixelized_image(
    image: np.ndarray,
    masks: List[Dict],
    pixel_size: int = DEFAULT_PIXEL_SIZE
) -> np.ndarray:
    """
    Create a pixelized image with mask.

    Args:
        image: Original image
        masks: List of mask data
        pixel_size: Pixel size for pixelization

    Returns:
        Pixelized image
    """

    final_result = image.copy()

    def pixelize(img: np.ndarray, mask: np.ndarray[np.uint8], pixel_size: int):
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        return np.where(mask[:, :, np.newaxis] > 0, pixelated, img)

    for info in masks:
        rle = info['segmentation']
        mask = decode_to_mask(rle)

        pixelated_segment = pixelize(final_result, mask, pixel_size)

        final_result = np.where(mask[:, :, np.newaxis] > 0, pixelated_segment, final_result)

    return final_result


def create_solid_color_mask_image(
    image: np.ndarray,
    masks: List[Dict],
    color_hex: str = DEFAULT_COLOR
) -> np.ndarray:
    """
    Create an image with solid color masks.

    Args:
        image: Original image
        masks: List of mask data
        color_hex: Hex color code

    Returns:
        Image with solid color masks
    """
    final_result = image.copy()

    def hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb[::-1]

    color_bgr = hex_to_bgr(color_hex)

    for info in masks:
        rle = info['segmentation']
        mask = decode_to_mask(rle)

        solid_color_mask = np.full(image.shape, color_bgr, dtype=np.uint8)

        final_result = np.where(mask[:, :, np.newaxis] > 0, solid_color_mask, final_result)

    return final_result


def insert_psd_layer(
    psd: PsdFile,
    image_data: np.ndarray,
    layer_name: str,
    blending_mode: BlendMode
) -> PsdFile:
    """
    Insert a layer into the PSD file using pytoshop

    Args:
        psd: PSD file object from the pytoshop
        image_data: Image data
        layer_name: Layer name
        blending_mode: Blending mode from pytoshop

    Returns:
        Updated PSD file object
    """

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


def save_psd(
    input_image_data: np.ndarray,
    layer_data: List,
    layer_names: List,
    blending_modes: List,
    output_path: str
):
    """
    Save the image with multiple layers as a PSD file

    Args:
        input_image_data: Original image data
        layer_data: List of images to be saved as layers
        layer_names: List of layer names
        blending_modes: List of blending modes
        output_path: Output path for the PSD file
    """

    psd_file = PsdFile(num_channels=3, height=input_image_data.shape[0], width=input_image_data.shape[1])
    psd_file.layer_and_mask_info.layer_info.layer_records.clear()

    for index, layer in enumerate(layer_data):
        psd_file = insert_psd_layer(psd_file, layer, layer_names[index], blending_modes[index])

    with open(output_path, 'wb') as output_file:
        psd_file.write(output_file)


def save_psd_with_masks(
    image: np.ndarray,
    masks: List[Dict],
    output_path: str
):
    """
    Save the psd file with masks data.

    Args:
        image: Original image
        masks: List of mask data
        output_path: Output path for the PSD file
    """
    original_layer = create_base_layer(image)
    mask_layers = create_mask_layers(image, masks)
    names = [f'Part {i}' for i in range(len(mask_layers))]
    modes = [BlendMode.normal] * (len(mask_layers)+1)
    save_psd(image, original_layer+mask_layers, ['Original_Image']+names, modes, output_path)


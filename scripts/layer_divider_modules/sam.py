from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import torch

from scripts.layer_divider_modules.mask_utils import *
from scripts.layer_divider_modules.model_downloader import *

from modules import safe


class SamInference:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(sam_model_path, "sam_vit_h_4b8939.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_generator = None

        # Tuable Parameters , All default values
        self.tunable_params = {
            'points_per_side': 32,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 1,
            'min_mask_region_area': 0
        }

    def set_mask_generator(self):
        if not is_sam_exist():
            print("\nLayer Divider Extension : No SAM model found, downloading ViT-H SAM model...")
            download_sam_model_url()
        print("\nLayer Divider Extension : applying configs to model..")

        try:
            torch.load = safe.unsafe_torch_load
            self.model = sam_model_registry["default"](checkpoint=self.model_path)
            self.model.to(device=self.device)
            torch.load = safe.load
        except Exception as e:
            print(f"Layer Divider Extension : Error while Loading SAM model! {e}")

        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=self.tunable_params['points_per_side'],
            pred_iou_thresh=self.tunable_params['pred_iou_thresh'],
            stability_score_thresh=self.tunable_params['stability_score_thresh'],
            crop_n_layers=self.tunable_params['crop_n_layers'],
            crop_n_points_downscale_factor=self.tunable_params['crop_n_points_downscale_factor'],
            min_mask_region_area=self.tunable_params['min_mask_region_area'],
            output_mode="coco_rle",
        )

    def generate_mask(self, image):
        return [self.mask_generator.generate(image)]

    def generate_mask_app(self, image, *params):
        tunable_params = {
            'points_per_side': int(params[0]),
            'pred_iou_thresh': float(params[1]),
            'stability_score_thresh': float(params[2]),
            'crop_n_layers': int(params[3]),
            'crop_n_points_downscale_factor': int(params[4]),
            'min_mask_region_area': int(params[5]),
        }

        if self.model is None or self.mask_generator is None or self.tunable_params != tunable_params:
            self.tunable_params = tunable_params
            self.set_mask_generator()

        masks = self.mask_generator.generate(image)
        save_psd_with_masks(image, masks)
        combined_image = create_mask_combined_images(image, masks)
        gallery = create_mask_gallery(image, masks)
        return [combined_image] + gallery
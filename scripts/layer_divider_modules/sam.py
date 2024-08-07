from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os

from scripts.layer_divider_modules.mask_utils import *
from scripts.layer_divider_modules.model_downloader import *
from scripts.layer_divider_modules.installation import base_dir

from modules import safe
# https://huggingface.co/spaces/SkalskiP/segment-anything-model-2/blob/main/utils/models.py
CONFIG_PATH = os.path.join(base_dir, "configs")
CONFIGS = {
    "sam2_hiera_tiny": os.path.join(CONFIG_PATH, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(CONFIG_PATH, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(CONFIG_PATH, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(CONFIG_PATH, "sam2_hiera_l.yaml"),
}


class SamInference:
    def __init__(self):
        self.model = None
        self.available_models = list(AVAILABLE_MODELS.keys())
        self.model_type = DEFAULT_MODEL_TYPE
        self.model_path = os.path.join(SAM_MODEL_PATH, AVAILABLE_MODELS[DEFAULT_MODEL_TYPE][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_generator = None
        self.image_predictor = None
        os.makedirs(SAM_MODEL_PATH, exist_ok=True)

        # Tunable Parameters , All default values by https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
        self.tunable_params = {
            "points_per_side": 64,
            "points_per_batch": 128,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.92,
            "stability_score_offset": 0.7,
            "crop_n_layers": 1,
            "box_nms_thresh": 0.7,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 25.0,
            "use_m2m": True,
        }

    def load_model(self):
        config = CONFIGS[self.model_type]
        model_path = os.path.join(SAM_MODEL_PATH, AVAILABLE_MODELS[self.model_type][0])

        if not is_sam_exist(self.model_type):
            print(f"\nLayer Divider Extension : No SAM2 model found, downloading {self.model_type} model...")
            download_sam_model_url(self.model_type)
        print("\nLayer Divider Extension : applying configs to model..")

        try:
            torch.load = safe.unsafe_torch_load
            self.model = build_sam2(config, model_path, device=self.device)
            self.image_predictor = SAM2ImagePredictor(sam_model=self.model)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                **self.tunable_params
            )
            torch.load = safe.load
        except Exception as e:
            print(f"Layer Divider Extension : Error while Loading SAM2 model! {e}")

    def generate_mask(self, image):
        return [self.mask_generator.generate(image)]

    def generate_mask_app(self, image, model_type, *params):
        tunable_params = {
            'points_per_side': int(params[0]),
            'points_per_batch': int(params[1]),
            'pred_iou_thresh': float(params[2]),
            'stability_score_thresh': float(params[3]),
            'stability_score_offset': float(params[4]),
            'crop_n_layers': int(params[5]),
            'box_nms_thresh': float(params[6]),
            'crop_n_points_downscale_factor': int(params[7]),
            'min_mask_region_area': int(params[8]),
            'use_m2m': bool(params[9])
        }

        if self.model is None or self.mask_generator is None or self.model_type != model_type or self.tunable_params != tunable_params:
            self.model_type = model_type
            self.tunable_params = tunable_params
            print(f"self.tunable_params: {self.tunable_params}")
            self.load_model()

        masks = self.mask_generator.generate(image)

        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_path = os.path.join(base_dir, "layer_divider_outputs", "psd", f"result-{timestamp}.psd")

        save_psd_with_masks(image, masks, output_path)
        combined_image = create_mask_combined_images(image, masks)
        gallery = create_mask_gallery(image, masks)
        return [combined_image] + gallery, output_path

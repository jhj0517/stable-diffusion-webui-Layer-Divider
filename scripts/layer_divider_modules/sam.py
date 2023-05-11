from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

from scripts.layer_divider_modules.mask_utils import *
from scripts.layer_divider_modules.model_downloader import *
from scripts.layer_divider_modules.installation import base_dir

from modules import safe


class SamInference:
    def __init__(self):
        self.model = None
        self.available_models = list(AVAILABLE_MODELS.keys())
        self.model_type = DEFAULT_MODEL_TYPE
        self.model_path = os.path.join(sam_model_path, AVAILABLE_MODELS[DEFAULT_MODEL_TYPE][0])
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
        if not is_sam_exist(self.model_type):
            print(f"\nLayer Divider Extension : No SAM model found, downloading {self.model_type} model...")
            download_sam_model_url(self.model_type)
        print("\nLayer Divider Extension : applying configs to model..")

        try:
            torch.load = safe.unsafe_torch_load
            self.model_path = os.path.join(sam_model_path, AVAILABLE_MODELS[self.model_type][0])
            self.model = sam_model_registry[self.model_type](checkpoint=self.model_path)
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

    def generate_mask_app(self, image, model_type, *params):
        tunable_params = {
            'points_per_side': int(params[0]),
            'pred_iou_thresh': float(params[1]),
            'stability_score_thresh': float(params[2]),
            'crop_n_layers': int(params[3]),
            'crop_n_points_downscale_factor': int(params[4]),
            'min_mask_region_area': int(params[5]),
        }

        if self.model is None or self.mask_generator is None or self.model_type != model_type or self.tunable_params != tunable_params:
            self.model_type = model_type
            self.tunable_params = tunable_params
            self.set_mask_generator()

        masks = self.mask_generator.generate(image)

        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_path = os.path.join(base_dir, "layer_divider_outputs", "psd", f"result-{timestamp}.psd")

        save_psd_with_masks(image, masks, output_path)
        combined_image = create_mask_combined_images(image, masks)
        gallery = create_mask_gallery(image, masks)
        return [combined_image] + gallery, output_path

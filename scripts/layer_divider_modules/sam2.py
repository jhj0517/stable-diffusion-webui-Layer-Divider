from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Dict, List, Optional, Tuple, Any
import torch
import os
from datetime import datetime
import numpy as np
import gradio as gr
import yaml

from scripts.layer_divider_modules.model_downloader import (
    AVAILABLE_MODELS, DEFAULT_MODEL_TYPE,
    is_sam_exist,
    download_sam_model_url
)
from scripts.layer_divider_modules.paths import (SAM2_MODEL_DIR, TEMP_OUT_DIR, TEMP_DIR, SAM2_MODEL_CONFIGS, OUTPUT_DIR,
                                                 SAM2_CONFIGS_DIR)
from scripts.layer_divider_modules.constants import (BOX_PROMPT_MODE, AUTOMATIC_MODE, COLOR_FILTER, PIXELIZE_FILTER, IMAGE_FILE_EXT)
from scripts.layer_divider_modules.mask_utils import (
    invert_masks,
    save_psd_with_masks,
    create_mask_combined_images,
    create_mask_gallery,
    create_mask_pixelized_image,
    create_solid_color_mask_image
)
from scripts.layer_divider_modules.video_utils import (get_frames_from_dir, create_video_from_frames, get_video_info, extract_frames,
                                 extract_sound, clean_temp_dir, clean_files_with_extension)
from scripts.layer_divider_modules.file_utils import save_image


class SamInference:
    def __init__(self,
                 model_dir: str = SAM2_MODEL_DIR,
                 output_dir: str = OUTPUT_DIR
                 ):
        self.model = None
        self.available_models = list(AVAILABLE_MODELS.keys())
        self.current_model_type = DEFAULT_MODEL_TYPE
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        self.mask_generator = None
        self.image_predictor = None
        self.video_predictor = None
        self.video_inference_state = None
        self.video_info = None

        default_hparam_config_path = os.path.join(SAM2_CONFIGS_DIR, "default_hparams.yaml")
        with open(default_hparam_config_path, 'r') as file:
            self.default_hparams = yaml.safe_load(file)

    def load_model(self,
                   model_type: Optional[str] = None,
                   load_video_predictor: bool = False):
        """
        Load the model from the model directory. If the model is not found, download it from the URL.

        Args:
            model_type (str): The model type to load.
            load_video_predictor (bool): Load the video predictor model.
        """
        if model_type is None:
            model_type = DEFAULT_MODEL_TYPE

        config_path = SAM2_MODEL_CONFIGS[model_type]
        config_dir, config_name = os.path.split(config_path)

        filename, url = AVAILABLE_MODELS[model_type]

        model_path = os.path.join(self.model_dir, filename)

        if not is_sam_exist(model_dir=self.model_dir, model_type=model_type):
            print(f"No SAM2 model found, downloading {model_type} model...")
            download_sam_model_url(model_dir=self.model_dir, model_type=model_type)
        print(f"Applying configs to {model_type} model..")

        if load_video_predictor:
            try:
                self.model = None
                self.video_predictor = build_sam2_video_predictor(
                    config_file=config_name,
                    ckpt_path=model_path,
                    device=self.device
                )
                return
            except Exception as e:
                print("Layer Divider: Error while loading SAM2 model for video predictor")

        try:
            self.video_predictor = None
            self.model = build_sam2(
                config_file=config_name,
                ckpt_path=model_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Layer Divider: Failed to load model") from e

    def init_video_inference_state(self,
                                   vid_input: str,
                                   model_type: Optional[str] = None):
        """
        Initialize the video inference state for the video predictor.

        Args:
            vid_input (str): The video frames directory.
            model_type (str): The model type to load.
        """
        if model_type is None:
            model_type = self.current_model_type

        if self.video_predictor is None or model_type != self.current_model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type, load_video_predictor=True)

        self.video_info = get_video_info(vid_input)
        frames_temp_dir = TEMP_DIR
        clean_temp_dir(frames_temp_dir)
        extract_frames(vid_input, frames_temp_dir)
        if self.video_info.has_sound:
            extract_sound(vid_input, frames_temp_dir)

        if self.video_inference_state is not None:
            self.video_predictor.reset_state(self.video_inference_state)
            self.video_inference_state = None

        self.video_inference_state = self.video_predictor.init_state(video_path=frames_temp_dir)

    def generate_mask(self,
                      image: np.ndarray,
                      model_type: str,
                      invert_mask: bool = False,
                      **params) -> List[Dict[str, Any]]:
        """
        Generate masks with Automatic segmentation. Default hyperparameters are in './configs/default_hparams.yaml.'

        Args:
            image (np.ndarray): The input image.
            model_type (str): The model type to load.
            invert_mask (bool): Invert the mask output - used for background masking.
            **params: The hyperparameters for the mask generator.

        Returns:
            List[Dict[str, Any]]: The auto-generated mask data.
        """

        if self.model is None or self.current_model_type != model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            **params
        )
        try:
            generated_masks = self.mask_generator.generate(image)
        except Exception as e:
            raise RuntimeError(f"Failed to generate masks") from e

        if invert_mask:
            generated_masks = [{'segmentation': invert_masks(mask['segmentation']),
                                'area': mask['area']} for mask in generated_masks]

        return generated_masks

    def predict_image(self,
                      image: np.ndarray,
                      model_type: str,
                      box: Optional[np.ndarray] = None,
                      point_coords: Optional[np.ndarray] = None,
                      point_labels: Optional[np.ndarray] = None,
                      invert_mask: bool = False,
                      **params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict image with prompt data.

        Args:
            image (np.ndarray): The input image.
            model_type (str): The model type to load.
            box (np.ndarray): The box prompt data.
            point_coords (np.ndarray): The point coordinates prompt data.
            point_labels (np.ndarray): The point labels prompt data.
            invert_mask (bool): Invert the mask output - used for background masking.
            **params: The hyperparameters for the mask generator.

        Returns:
            np.ndarray: The predicted masks output in CxHxW format.
            np.ndarray: Array of scores for each mask.
            np.ndarray: Array of logits in CxHxW format.
        """
        if self.model is None or self.current_model_type != model_type:
            self.current_model_type = model_type
            self.load_model(model_type=model_type)
        self.image_predictor = SAM2ImagePredictor(sam_model=self.model)
        self.image_predictor.set_image(image)

        try:
            masks, scores, logits = self.image_predictor.predict(
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=params["multimask_output"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to predict image with prompt") from e

        if invert_mask:
            masks = invert_masks(masks)

        return masks, scores, logits

    def add_prediction_to_frame(self,
                                frame_idx: int,
                                obj_id: int,
                                inference_state: Optional[Dict] = None,
                                points: Optional[np.ndarray] = None,
                                labels: Optional[np.ndarray] = None,
                                box: Optional[np.ndarray] = None) -> Tuple[int, int, torch.Tensor]:
        """
        Add prediction to the current video inference state. inference state must be initialized before calling this method.

        Args:
            frame_idx (int): The frame index of the video.
            obj_id (int): The object id for the frame.
            inference_state (Dict): The inference state for the video predictor.
            points (np.ndarray): The point coordinates prompt data.
            labels (np.ndarray): The point labels prompt data.
            box (np.ndarray): The box prompt data.

        Returns:
            int: The frame index of the corresponding prediction.
            int: The object id of the corresponding prediction.
            torch.Tensor: The mask logits output in CxHxW format.
        """

        if (self.video_predictor is None or
                inference_state is None and self.video_inference_state is None):
            print("Error while predicting frame from video, load video predictor first")

        if inference_state is None:
            inference_state = self.video_inference_state

        try:
            out_frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box
            )
        except Exception as e:
            raise RuntimeError(f"Failed to predicting frame with prompt") from e

        return out_frame_idx, out_obj_ids, out_mask_logits

    def propagate_in_video(self,
                           inference_state: Optional[Dict] = None,):
        """
        Propagate in the video with the tracked predictions for each frame. Currently only supports
        single frame tracking.

        Args:
            inference_state (Dict): The inference state for the video predictor. Use self.video_inference_state if None.

        Returns:
            Dict: The video segments with the image and mask data. It has frame index as each key and each key has
                "image" and "mask" data. "image" key contains the path of the original image file and "mask" key contains
                the np.ndarray mask output.
        """
        if inference_state is None and self.video_inference_state is None:
            print("Error while propagating in video, load video predictor first")

        if inference_state is None:
            inference_state = self.video_inference_state

        video_segments = {}

        try:
            generator = self.video_predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0
            )
            images = get_frames_from_dir(vid_dir=TEMP_DIR, as_numpy=True)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                for out_frame_idx, out_obj_ids, out_mask_logits in generator:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    video_segments[out_frame_idx] = {
                        "image": images[out_frame_idx],
                        "mask": mask
                    }
        except Exception as e:
            raise RuntimeError(f"Failed to propagate in video") from e

        return video_segments

    def add_filter_to_preview(self,
                              image_prompt_input_data: Dict,
                              filter_mode: str,
                              frame_idx: int,
                              pixel_size: Optional[int] = None,
                              color_hex: Optional[str] = None,
                              invert_mask: bool = False
                              ):
        """
        Add filter to the preview image with the prompt data. Specially made for gradio app.
        It adds prediction tracking to the self.video_inference_state and returns the filtered image.

        Args:
            image_prompt_input_data (Dict): The image prompt data.
            filter_mode (str): The filter mode to apply. ["Solid Color", "Pixelize"]
            frame_idx (int): The frame index of the video.
            pixel_size (int): The pixel size for the pixelize filter.
            color_hex (str): The color hex code for the solid color filter.
            invert_mask (bool): Invert the mask output - used for background masking.

        Returns:
            np.ndarray: The filtered image output.
        """
        if self.video_predictor is None or self.video_inference_state is None:
            raise f"Error while adding filter to preview"

        if not image_prompt_input_data["points"]:
            error_message = ("No prompt data provided. If this is an incorrect flag, "
                             "Please press the eraser button (on the image prompter) and add your prompts again.")
            raise gr.Error(error_message, duration=20)

        image, prompt = image_prompt_input_data["image"], image_prompt_input_data["points"]
        image = np.array(image.convert("RGB"))

        point_labels, point_coords, box = self.handle_prompt_data(prompt)
        obj_id = frame_idx

        self.video_predictor.reset_state(self.video_inference_state)
        idx, scores, logits = self.add_prediction_to_frame(
            frame_idx=frame_idx,
            obj_id=obj_id,
            inference_state=self.video_inference_state,
            points=point_coords,
            labels=point_labels,
            box=box
        )
        masks = (logits[0] > 0.0).cpu().numpy()
        if invert_mask:
            masks = invert_masks(masks)

        generated_masks = self.format_to_auto_result(masks)

        if filter_mode == COLOR_FILTER:
            image = create_solid_color_mask_image(image, generated_masks, color_hex)

        elif filter_mode == PIXELIZE_FILTER:
            image = create_mask_pixelized_image(image, generated_masks, pixel_size)

        return image

    def create_filtered_video(self,
                              image_prompt_input_data: Dict,
                              filter_mode: str,
                              frame_idx: int,
                              pixel_size: Optional[int] = None,
                              color_hex: Optional[str] = None,
                              invert_mask: bool = False
                              ):
        """
        Create a whole filtered video with video_inference_state. Currently only one frame tracking is supported.
        This needs FFmpeg to run. Returns two output path because of the gradio app.

        Args:
            image_prompt_input_data (Dict): The image prompt data.
            filter_mode (str): The filter mode to apply. ["Solid Color", "Pixelize"]
            frame_idx (int): The frame index of the video.
            pixel_size (int): The pixel size for the pixelize filter.
            color_hex (str): The color hex code for the solid color filter.
            invert_mask (bool): Invert the mask output - used for background masking.

        Returns:
            str: The output video path.
            str: The output video path.
        """

        if self.video_predictor is None or self.video_inference_state is None:
            raise RuntimeError("Error while adding filter to preview")

        if not image_prompt_input_data["points"]:
            error_message = ("No prompt data provided. If this is an incorrect flag, "
                             "Please press the eraser button (on the image prompter) and add your prompts again.")
            raise gr.Error(error_message, duration=20)
        output_dir = os.path.join(self.output_dir, "filter")

        clean_files_with_extension(TEMP_OUT_DIR, IMAGE_FILE_EXT)
        self.video_predictor.reset_state(self.video_inference_state)

        prompt_frame_image, prompt = image_prompt_input_data["image"], image_prompt_input_data["points"]

        point_labels, point_coords, box = self.handle_prompt_data(prompt)
        obj_id = frame_idx

        idx, scores, logits = self.add_prediction_to_frame(
            frame_idx=frame_idx,
            obj_id=obj_id,
            inference_state=self.video_inference_state,
            points=point_coords,
            labels=point_labels,
            box=box,
        )

        video_segments = self.propagate_in_video(inference_state=self.video_inference_state)
        for frame_index, info in video_segments.items():
            orig_image, masks = info["image"], info["mask"]
            if invert_mask:
                masks = invert_masks(masks)
            masks = self.format_to_auto_result(masks)

            if filter_mode == COLOR_FILTER:
                filtered_image = create_solid_color_mask_image(orig_image, masks, color_hex)

            elif filter_mode == PIXELIZE_FILTER:
                filtered_image = create_mask_pixelized_image(orig_image, masks, pixel_size)

            save_image(image=filtered_image, output_dir=TEMP_OUT_DIR)

        if len(video_segments) == 1:
            out_image = save_image(image=filtered_image, output_dir=output_dir)
            return None, out_image

        out_video = create_video_from_frames(
            frames_dir=TEMP_OUT_DIR,
            frame_rate=self.video_info.frame_rate,
            output_dir=output_dir,
        )

        return out_video, out_video

    def divide_layer(self,
                     image_input: np.ndarray,
                     image_prompt_input_data: Dict,
                     input_mode: str,
                     model_type: str,
                     invert_mask: bool = False,
                     *params):
        """
        Divide the layer with the given prompt data and save psd file.

        Args:
            image_input (np.ndarray): The input image.
            image_prompt_input_data (Dict): The image prompt data.
            input_mode (str): The input mode for the image prompt data. ["Automatic", "Box Prompt"]
            model_type (str): The model type to load.
            invert_mask (bool): Invert the mask output.
            *params: The hyperparameters for the mask generator.

        Returns:
            List[np.ndarray]: List of images by predicted masks.
            str: The output path of the psd file.
        """

        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name = f"result-{timestamp}.psd"
        output_path = os.path.join(self.output_dir, "psd", output_file_name)
        # Pre-processed gradio components
        hparams = {
            'points_per_side': int(params[0]),
            'points_per_batch': int(params[1]),
            'pred_iou_thresh': float(params[2]),
            'stability_score_thresh': float(params[3]),
            'stability_score_offset': float(params[4]),
            'crop_n_layers': int(params[5]),
            'box_nms_thresh': float(params[6]),
            'crop_n_points_downscale_factor': int(params[7]),
            'min_mask_region_area': int(params[8]),
            'use_m2m': bool(params[9]),
            'multimask_output': bool(params[10])
        }

        if input_mode == AUTOMATIC_MODE:
            image = image_input

            generated_masks = self.generate_mask(
                image=image,
                model_type=model_type,
                invert_mask=invert_mask,
                **hparams
            )

        elif input_mode == BOX_PROMPT_MODE:
            image = image_prompt_input_data["image"]
            image = np.array(image.convert("RGB"))
            prompt = image_prompt_input_data["points"]
            if len(prompt) == 0:
                return [image], []

            point_labels, point_coords, box = self.handle_prompt_data(prompt)

            predicted_masks, scores, logits = self.predict_image(
                image=image,
                model_type=model_type,
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=hparams["multimask_output"],
                invert_mask=invert_mask
            )
            generated_masks = self.format_to_auto_result(predicted_masks)

        save_psd_with_masks(image, generated_masks, output_path)
        mask_combined_image = create_mask_combined_images(image, generated_masks)
        gallery = create_mask_gallery(image, generated_masks)
        gallery = [mask_combined_image] + gallery

        return gallery, output_path

    @staticmethod
    def format_to_auto_result(
        masks: np.ndarray
    ):
        """Format the masks to auto result format for convenience."""
        place_holder = 0
        if len(masks.shape) <= 3:
            masks = np.expand_dims(masks, axis=0)
        result = [{"segmentation": mask[0], "area": place_holder} for mask in masks]
        return result

    @staticmethod
    def handle_prompt_data(
        prompt_data: List
    ):
        """
        Handle data from ImageInputPrompter.

        Args:
            prompt_data (Dict): A dictionary containing the 'prompt' key with a list of prompts.

        Returns:
            point_labels (List): list of points labels.
            point_coords (List): list of points coords.
            box (List): list of box datas.
        """
        point_labels, point_coords, box = [], [], []

        for x1, y1, left_click_indicator, x2, y2, point_indicator in prompt_data:
            is_point = point_indicator == 4.0
            if is_point:
                point_labels.append(left_click_indicator)
                point_coords.append([x1, y1])
            else:
                box.append([x1, y1, x2, y2])

        point_labels = np.array(point_labels) if point_labels else None
        point_coords = np.array(point_coords) if point_coords else None
        box = np.array(box) if box else None

        return point_labels, point_coords, box

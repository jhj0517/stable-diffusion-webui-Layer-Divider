from scripts.layer_divider_modules.installation import *
install_sam()
from scripts.layer_divider_modules import sam2
from scripts.layer_divider_modules.ui_utils import *
from scripts.layer_divider_modules.html_constants import *
from scripts.layer_divider_modules.video_utils import get_frames_from_dir
from scripts.layer_divider_modules.default_hparams import DEFAULT_HPARAMS
from scripts.layer_divider_modules.model_downloader import DEFAULT_MODEL_TYPE
from scripts.layer_divider_modules.paths import (SAM2_CONFIGS_DIR, TEMP_DIR, TEMP_OUT_DIR, OUTPUT_DIR,
                                                 OUTPUT_FILTER_DIR, OUTPUT_PSD_DIR, init_paths)
from scripts.layer_divider_modules.constants import (AUTOMATIC_MODE, BOX_PROMPT_MODE, COLOR_FILTER, PIXELIZE_FILTER,
                                                     IMAGE_FILE_EXT, VIDEO_FILE_EXT, DEFAULT_COLOR, DEFAULT_PIXEL_SIZE)

import gradio as gr
from gradio_image_prompter import ImagePrompter
import os
import yaml
from typing import Dict, Optional

from modules import scripts, script_callbacks

sam_inf = sam2.SamInference()

def mask_generation_parameters(hparams: Optional[Dict] = None):
    if hparams is None:
        default_hparams = DEFAULT_HPARAMS
        hparams = default_hparams["mask_hparams"]
    mask_components = [
        gr.Number(label="points_per_side ", value=hparams["points_per_side"], interactive=True),
        gr.Number(label="points_per_batch ", value=hparams["points_per_batch"], interactive=True),
        gr.Slider(label="pred_iou_thresh ", value=hparams["pred_iou_thresh"], minimum=0, maximum=1,
                  interactive=True),
        gr.Slider(label="stability_score_thresh ", value=hparams["stability_score_thresh"], minimum=0,
                  maximum=1, interactive=True),
        gr.Slider(label="stability_score_offset ", value=hparams["stability_score_offset"], minimum=0,
                  maximum=1),
        gr.Number(label="crop_n_layers ", value=hparams["crop_n_layers"]),
        gr.Slider(label="box_nms_thresh ", value=hparams["box_nms_thresh"], minimum=0, maximum=1),
        gr.Number(label="crop_n_points_downscale_factor ", value=hparams["crop_n_points_downscale_factor"]),
        gr.Number(label="min_mask_region_area ", value=hparams["min_mask_region_area"]),
        gr.Checkbox(label="use_m2m ", value=hparams["use_m2m"])
    ]
    return mask_components


def on_mode_change(mode: str):
    return [
        gr.Image(visible=mode == AUTOMATIC_MODE),
        ImagePrompter(visible=mode == BOX_PROMPT_MODE),
        gr.Accordion(visible=mode == AUTOMATIC_MODE),
    ]


def on_filter_mode_change(mode: str):
    return [
        gr.ColorPicker(visible=mode == COLOR_FILTER),
        gr.Number(visible=mode == PIXELIZE_FILTER)
    ]


def on_video_model_change(model_type: str,
                          vid_input: str):
    global sam_inf
    sam_inf.init_video_inference_state(vid_input=vid_input, model_type=model_type)
    frames = get_frames_from_dir(vid_dir=TEMP_DIR)
    initial_frame, max_frame_index = frames[0], (len(frames)-1)
    return [
        ImagePrompter(label="Prompt image with Box & Point", value=initial_frame),
        gr.Slider(label="Frame Index", value=0, interactive=True, step=1, minimum=0, maximum=max_frame_index)
    ]


def on_frame_change(frame_idx: int):
    temp_dir = TEMP_DIR
    frames = get_frames_from_dir(vid_dir=temp_dir)
    selected_frame = frames[frame_idx]
    return ImagePrompter(label=f"Prompt image with Box & Point", value=selected_frame)


def on_prompt_change(prompt: Dict):
    image, points = prompt["image"], prompt["points"]
    return gr.Image(label="Preview", value=image)


def add_tab():
    global sam_inf
    default_filter = COLOR_FILTER
    default_color = DEFAULT_COLOR
    default_pixel_size = DEFAULT_PIXEL_SIZE
    filter_modes = [COLOR_FILTER, PIXELIZE_FILTER]
    image_modes = [AUTOMATIC_MODE, BOX_PROMPT_MODE]
    default_mode = BOX_PROMPT_MODE
    _mask_hparams = DEFAULT_HPARAMS["mask_hparams"]

    init_paths()

    with gr.Blocks():
        with gr.Tabs() as tab:
            with gr.TabItem("Filter to Video"):
                with gr.Column():
                    file_vid_input = gr.File(label="Upload Input Video", file_types=IMAGE_FILE_EXT + VIDEO_FILE_EXT)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=9):
                            with gr.Row():
                                vid_frame_prompter = ImagePrompter(label="Prompt image with Box & Point", type='pil',
                                                                   interactive=True, scale=5)
                                img_preview = gr.Image(label="Preview", interactive=False, scale=5)

                            sld_frame_selector = gr.Slider(label="Frame Index", interactive=False)

                        with gr.Column(scale=1):
                            dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                                    choices=sam_inf.available_models)
                            dd_filter_mode = gr.Dropdown(label="Filter Modes", interactive=True,
                                                         value=default_filter,
                                                         choices=filter_modes)
                            cp_color_picker = gr.ColorPicker(label="Solid Color", interactive=True,
                                                             visible=default_filter == COLOR_FILTER,
                                                             value=default_color)
                            nb_pixel_size = gr.Number(label="Pixel Size", interactive=True, minimum=1,
                                                      visible=default_filter == PIXELIZE_FILTER,
                                                      value=default_pixel_size)
                            cb_invert_mask = gr.Checkbox(label="invert mask", value=_mask_hparams["invert_mask"])
                            btn_generate_preview = gr.Button("GENERATE PREVIEW")

                with gr.Row():
                    btn_generate = gr.Button("GENERATE", variant="primary")
                with gr.Row():
                    vid_output = gr.Video(label="Output Video", interactive=False)
                    with gr.Column():
                        output_file = gr.File(label="Downloadable Output File", scale=9)
                        btn_open_folder = gr.Button("📁\nOpen Output folder", scale=1)

                file_vid_input.change(fn=on_video_model_change,
                                      inputs=[dd_models, file_vid_input],
                                      outputs=[vid_frame_prompter, sld_frame_selector])
                dd_models.change(fn=on_video_model_change,
                                 inputs=[dd_models, file_vid_input],
                                 outputs=[vid_frame_prompter, sld_frame_selector])
                sld_frame_selector.change(fn=on_frame_change,
                                          inputs=[sld_frame_selector],
                                          outputs=[vid_frame_prompter], )
                dd_filter_mode.change(fn=on_filter_mode_change,
                                      inputs=[dd_filter_mode],
                                      outputs=[cp_color_picker,
                                               nb_pixel_size])

                preview_params = [vid_frame_prompter, dd_filter_mode, sld_frame_selector, nb_pixel_size,
                                  cp_color_picker, cb_invert_mask]
                btn_generate_preview.click(fn=sam_inf.add_filter_to_preview,
                                           inputs=preview_params,
                                           outputs=[img_preview])
                btn_generate.click(fn=sam_inf.create_filtered_video,
                                   inputs=preview_params,
                                   outputs=[vid_output, output_file])
                btn_open_folder.click(fn=lambda: open_folder(OUTPUT_FILTER_DIR),
                                      inputs=None,
                                      outputs=None)

            with gr.TabItem("Layer Divider"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_input = gr.Image(label="Input image here", visible=default_mode == AUTOMATIC_MODE)
                        img_input_prompter = ImagePrompter(label="Prompt image with Box & Point", type='pil',
                                                           visible=default_mode == BOX_PROMPT_MODE)

                    with gr.Column(scale=5):
                        dd_input_modes = gr.Dropdown(label="Image Input Mode", value=default_mode,
                                                     choices=image_modes)
                        dd_models = gr.Dropdown(label="Model", value=DEFAULT_MODEL_TYPE,
                                                choices=sam_inf.available_models)
                        cb_invert_mask = gr.Checkbox(label="invert mask", value=_mask_hparams["invert_mask"])

                        with gr.Accordion("Mask Parameters", open=False,
                                          visible=default_mode == AUTOMATIC_MODE) as acc_mask_hparams:
                            mask_hparams_component = mask_generation_parameters(_mask_hparams)

                        cb_multimask_output = gr.Checkbox(label="multimask_output", value=_mask_hparams["multimask_output"])

                with gr.Row():
                    btn_generate = gr.Button("GENERATE", variant="primary")
                with gr.Row():
                    gallery_output = gr.Gallery(label="Output images will be shown here")
                    with gr.Column():
                        output_file = gr.File(label="Generated psd file", scale=9)
                        btn_open_folder = gr.Button("📁\nOpen PSD folder", scale=1)

                input_params = [img_input, img_input_prompter, dd_input_modes, dd_models, cb_invert_mask]
                mask_hparams = mask_hparams_component + [cb_multimask_output]
                input_params += mask_hparams

                btn_generate.click(fn=sam_inf.divide_layer,
                                   inputs=input_params, outputs=[gallery_output, output_file])
                btn_open_folder.click(fn=lambda: open_folder(OUTPUT_PSD_DIR),
                                      inputs=None, outputs=None)
                dd_input_modes.change(fn=on_mode_change,
                                      inputs=[dd_input_modes],
                                      outputs=[img_input, img_input_prompter, acc_mask_hparams])

        return [(tab, "Layer_Divider", "layer_divider")]


def on_unload():
    global sam_inf
    sam_inf = None


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_script_unloaded(on_unload)

# Layer-Divider
This is an implementaion of the [SAM](https://github.com/facebookresearch/segment-anything) (Segment-Anything Model) within the SD WebUI.

Divide layers in the SD WebUI and save them as PSD files.

![screenshot1](https://raw.githubusercontent.com/jhj0517/stable-diffusion-webui-Layer-Divider/master/screenshot.png)

![screenshot2](https://raw.githubusercontent.com/jhj0517/stable-diffusion-webui-Layer-Divider/master/screenshot2.png)

If you want a dedicated WebUI specifically for this, rather than as an extension, please visit this [repository](https://github.com/jhj0517/Layer-Divider-WebUI)

# Installation
`git clone https://github.com/jhj0517/stable-diffusion-webui-Layer-Divider.git` to your stable-diffusion-webui extensions folder.

or alternatively, download and unzip the repository in your extensions folder!


# How to use
Adjust the parameters and click "Generate". The output will be displayed below, and a PSD file will be saved in the `extensions\stable-diffusion-webui-layer-divider\layer_divider_outputs\psd` folder.

## Explanation of Parameters

| Parameter                      | Description                                                                                                                                                                                                                                                                              |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| points_per_side                | The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.                                                                                                            |
| pred_iou_thresh                | A filtering threshold in [0,1], using the model's predicted mask quality.                                                                                                                                                                                                               |
| stability_score_thresh         | A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.                                                                                                                                             |
| crops_n_layers                 | If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.                                                                                                                                |
| crop_n_points_downscale_factor | The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.                                                                                                                                                                                 |
| min_mask_region_area           | If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.                                                                                                                                  |

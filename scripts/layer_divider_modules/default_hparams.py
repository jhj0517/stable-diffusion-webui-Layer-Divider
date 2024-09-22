DEFAULT_HPARAMS = {
    "mask_hparams": {
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
        "multimask_output": True,
        "invert_mask": False
    }
}
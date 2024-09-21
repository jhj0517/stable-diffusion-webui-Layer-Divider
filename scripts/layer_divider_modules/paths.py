import os

from modules.paths import models_path

LAYER_DIVIDER_EXTENSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM2_CONFIGS_DIR = os.path.join(LAYER_DIVIDER_EXTENSION_DIR, "configs")
SAM2_MODEL_DIR = os.path.join(models_path, "sam2")
SAM2_MODEL_CONFIGS = {
    "sam2_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_l.yaml"),
}
OUTPUT_DIR = os.path.join(LAYER_DIVIDER_EXTENSION_DIR, "outputs")
OUTPUT_PSD_DIR = os.path.join(OUTPUT_DIR, "psd")
OUTPUT_FILTER_DIR = os.path.join(OUTPUT_DIR, "filter")
TEMP_DIR = os.path.join(LAYER_DIVIDER_EXTENSION_DIR, "temp")
TEMP_OUT_DIR = os.path.join(TEMP_DIR, "out")

for dir_path in [SAM2_MODEL_DIR,
                 SAM2_CONFIGS_DIR,
                 OUTPUT_DIR,
                 OUTPUT_PSD_DIR,
                 OUTPUT_FILTER_DIR,
                 TEMP_DIR,
                 TEMP_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

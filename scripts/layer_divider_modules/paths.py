import os

LAYER_DIVIDER_EXTENSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM2_CONFIGS_DIR = os.path.join(LAYER_DIVIDER_EXTENSION_DIR, "configs")
SAM2_MODEL_CONFIGS = {
    "sam2_hiera_tiny": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_t.yaml"),
    "sam2_hiera_small": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_s.yaml"),
    "sam2_hiera_base_plus": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_b+.yaml"),
    "sam2_hiera_large": os.path.join(SAM2_CONFIGS_DIR, "sam2_hiera_l.yaml"),
}
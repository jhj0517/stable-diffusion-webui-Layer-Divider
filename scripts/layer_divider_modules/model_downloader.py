import os
from typing import Optional

from modules import modelloader
from modules.sd_models import model_hash
from modules import shared

from scripts.layer_divider_modules.paths import SAM2_MODEL_DIR

DEFAULT_MODEL_TYPE = "sam2_hiera_large"

AVAILABLE_MODELS = {
    "sam2_hiera_tiny": ["sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"],
    "sam2_hiera_small": ["sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"],
    "sam2_hiera_base_plus": ["sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"],
    "sam2_hiera_large": ["sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"],
}


def list_models(model_path):
    model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])

    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    models = []
    for filename in model_list:
        h = model_hash(filename)
        title, short_model_name = modeltitle(filename, h)
        models.append(title)

    return models


def download_sam_model_url(model_type: str,
                           model_dir: str = SAM2_MODEL_DIR):
    shared.state.textinfo = "Downloading SAM model...."
    filename, url = AVAILABLE_MODELS[model_type]
    load_file_from_url(url=url, model_dir=model_dir)
    shared.state.textinfo = ""


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: Optional[str] = None,
) -> str:
    from urllib.parse import urlparse
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def is_sam_exist(
    model_type: str,
    model_dir: Optional[str] = None
):
    if model_dir is None:
        model_dir = SAM2_MODEL_DIR
    filename, url = AVAILABLE_MODELS[model_type]
    model_path = os.path.join(model_dir, filename)
    return os.path.exists(model_path)

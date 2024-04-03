import os
from typing import Optional

from modules.paths import models_path
from modules import modelloader
from modules.sd_models import model_hash
from modules import shared

sam_model_path = os.path.join(models_path, "sam")

DEFAULT_MODEL_TYPE = "vit_h"

AVAILABLE_MODELS = {
    "vit_h": ["sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"],
    "vit_l": ["sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"],
    "vit_b": ["sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"],
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


def download_sam_model_url(model_type):
    shared.state.textinfo = "Downloading SAM model...."
    load_file_from_url(url=AVAILABLE_MODELS[model_type][1], model_dir=sam_model_path)
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


def is_sam_exist(model_type):
    model_path = os.path.join(sam_model_path, AVAILABLE_MODELS[model_type][0])
    if not os.path.exists(model_path):
        return False
    else:
        return True

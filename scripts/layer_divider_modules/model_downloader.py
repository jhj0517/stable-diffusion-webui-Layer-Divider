import os

from basicsr.utils.download_util import load_file_from_url
from modules.paths import models_path
from modules import modelloader
from modules.sd_models import model_hash
from modules import shared

sam_model_path = os.path.join(models_path, "sam")

AVAILABLE_MODELS = {
    "ViT-H SAM model": ["sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"],
    "ViT-L SAM model": ["sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"],
    "ViT-B SAM model": ["sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"],
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


def download_sam_model_url():
    shared.state.textinfo = "Downloading SAM model...."
    load_file_from_url(AVAILABLE_MODELS["ViT-H SAM model"][1], sam_model_path)
    shared.state.textinfo = ""


def is_sam_exist():
    if len(list_models(sam_model_path)) == 0:
        return False
    else:
        return True

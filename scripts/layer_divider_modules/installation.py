import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def install_sam():
    req_file_path = os.path.join(base_dir, 'scripts', 'layer_divider_requirements.txt')
    from launch import is_installed, run
    if not is_installed("segment_anything") or not is_installed("pytoshop"):
        python = sys.executable
        run(f'"{python}" -m pip install -U -r "{req_file_path}"',
            desc="Layer Divider Extension: Insatlling requirements..",
            errdesc=f"\n\nCouldn't install requirements, the python is \n{python}\n\n")
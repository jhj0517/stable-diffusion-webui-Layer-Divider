import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def install_sam():
    req_file_path = os.path.join(base_dir, 'layer_divider_requirements.txt')
    from launch import is_installed, run_pip
    with open(req_file_path) as file:
        for package in file:
            package = package.strip()
            if not is_installed(package):
                run_pip(
                    f"install {package}", f"Layer Divider Extension: Installing  {package}")
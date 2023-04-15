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
    install_packbits_so_file()

def install_packbits_so_file():
    SO_FILES = {
        "packbits.cpython-36m-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-36m-darwin.so?raw=true",
        "packbits.cpython-37m-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-37m-darwin.so?raw=true",
        "packbits.cpython-38-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-38-darwin.so?raw=true",
        "packbits.cpython-39-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-39-darwin.so?raw=true",
        "packbits.cpython-310-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-310-darwin.so?raw=true",
        "packbits.cpython-311-darwin.so": "https://github.com/jhj0517/pytoshop_packbits_extension/blob/master/packbits_so/packbits.cpython-311-darwin.so?raw=true",
    }

    import sys
    import platform
    import urllib.request

    if platform.system() == "Darwin":
        sd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        pytoshop_path = os.path.join(sd_path, 'venv', 'Lib', 'site-packages', 'pytoshop')
        if not os.path.exists(pytoshop_path):
            print("pytoshop is not installed:", pytoshop_path)
            return

        so_file_name = f"packbits.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so"
        so_file_path = os.path.join(pytoshop_path, so_file_name)

        if not os.path.exists(so_file_path):
            print(f"packbits so file is not detected. downloading {so_file_name}")
            urllib.request.urlretrieve(SO_FILES[so_file_name], so_file_path)
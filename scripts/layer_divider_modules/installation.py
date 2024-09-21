import os
from setuptools.command.build_ext import build_ext
import distutils.sysconfig


def get_site_packages_path():
    sp_path = distutils.sysconfig.get_python_lib()
    return sp_path


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sd_path = os.path.abspath(os.path.join(base_dir, '..', '..'))
pytoshop_path = os.path.join(get_site_packages_path(), 'pytoshop')


def install_sam():
    req_file_path = os.path.join(base_dir, 'layer_divider_requirements.txt')
    from launch import is_installed, run_pip, run
    with open(req_file_path) as file:
        for package in file:
            if package.startswith("#"):
                continue

            package_name = package.strip()
            if "==" in package_name:
                package_name, version = package_name.split("==")

            if not is_installed(package_name):
                # run_pip(f"install {package}", f"Layer Divider Extension: Installing  {package}")
                run(f"python -m pip install -U {package}", f"Layer Divider Extension: Installing  {package}")


def build_packbits():
    # To resolve packbits bug. Instead of using this,
    # Follow README instruction : https://github.com/jhj0517/stable-diffusion-webui-Layer-Divider?tab=readme-ov-file#notice

    from setuptools import setup
    from setuptools.extension import Extension

    try:
        from Cython.Build import cythonize
    except ImportError:
        extensions = []
    else:
        extensions = cythonize([
            Extension(
                f"pytoshop.packbits",
                [os.path.join(pytoshop_path, "packbits.pyx")],
            )
        ])

    setup(
        name="pytoshop-packbits",
        ext_modules=extensions,
        cmdclass={'build_ext': CustomBuildExtCommand},
        script_args=['build_ext'],
    )


class CustomBuildExtCommand(build_ext):
    def get_ext_fullpath(self, ext_name):
        ext_path = ext_name.split('.')
        ext_suffix = self.get_ext_filename(ext_path[-1])
        filepath = os.path.join(pytoshop_path, 'packbits_temp' + ext_suffix)
        return filepath

    def run(self):
        # I don't know why, but the extension file is not being generated with the normal file name, so this needs to be done
        build_ext.run(self)

        for filename in os.listdir(pytoshop_path):
            if filename.startswith('packbits_temp'):
                new_filename = filename.replace('packbits_temp', '', 1)
                src = os.path.join(pytoshop_path, filename)
                dst = os.path.join(pytoshop_path, new_filename)

                os.rename(src, dst)
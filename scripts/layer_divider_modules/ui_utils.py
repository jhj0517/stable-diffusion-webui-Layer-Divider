import os


def open_folder(folder_path):
    if os.path.exists(folder_path):
        os.system(f'start "" "{folder_path}"')
    else:
        print(f"The folder '{folder_path}' does not exist.")
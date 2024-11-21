import subprocess
import os
from typing import List, Optional, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass
import re

from scripts.layer_divider_modules.constants import SOUND_FILE_EXT, VIDEO_FILE_EXT, IMAGE_FILE_EXT
from scripts.layer_divider_modules.paths import TEMP_DIR, TEMP_OUT_DIR


@dataclass
class VideoInfo:
    num_frames: Optional[int] = None
    frame_rate: Optional[int] = None
    duration: Optional[float] = None
    has_sound: Optional[bool] = None
    codec: Optional[str] = None


def extract_frames(
    vid_input: str,
    output_temp_dir: str = TEMP_DIR,
    start_number: int = 0
):
    """
    Extract frames as jpg files and save them into output_temp_dir. This needs FFmpeg installed.
    """
    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "%05d.jpg")

    command = [
        'ffmpeg',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-qscale:v', '2',
        '-vf', f'scale=iw:ih',
        '-start_number', str(start_number),
        f'{output_path}'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"An error occurred: {str(e)}")

    return get_frames_from_dir(output_temp_dir)


def extract_sound(
    vid_input: str,
    output_temp_dir: str = TEMP_DIR,
):
    """
    Extract audio from a video file and save it as a separate sound file. This needs FFmpeg installed.
    """
    os.makedirs(output_temp_dir, exist_ok=True)
    output_path = os.path.join(output_temp_dir, "sound.mp3")

    command = [
        'ffmpeg',
        '-y',  # Enable overwriting
        '-i', vid_input,
        '-vn',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred while extracting sound from the video")

    return output_path


def get_video_info(vid_input: str) -> VideoInfo:
    """
    Extract video information using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-i', vid_input,
        '-map', '0:v:0',
        '-c', 'copy',
        '-f', 'null',
        '-'
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', errors='replace', check=True)
        output = result.stderr

        num_frames = None
        frame_rate = None
        duration = None
        has_sound = False
        codec = None

        for line in output.splitlines():
            if 'Stream #0:0' in line and 'Video:' in line:
                fps_match = re.search(r'(\d+(?:\.\d+)?) fps', line)
                if fps_match:
                    frame_rate = float(fps_match.group(1))

                codec_match = re.search(r'Video: (\w+)', line)
                if codec_match:
                    codec = codec_match.group(1)

            elif 'Duration:' in line:
                duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                if duration_match:
                    h, m, s = map(float, duration_match.groups())
                    duration = h * 3600 + m * 60 + s

            elif 'Stream' in line and 'Audio:' in line:
                has_sound = True

        if frame_rate and duration:
            num_frames = int(frame_rate * duration)

        return VideoInfo(
            num_frames=num_frames,
            frame_rate=frame_rate,
            duration=duration,
            has_sound=has_sound,
            codec=codec
        )

    except subprocess.CalledProcessError as e:
        print("Error occurred while getting info from the video")
        return VideoInfo()


def create_video_from_frames(
    frames_dir: str,
    frame_rate: Optional[int] = None,
    sound_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Create a video from frames and save it to the output_path. This needs FFmpeg installed.
    """
    if not os.path.exists(frames_dir):
        raise "frames_dir does not exist"

    if output_dir is None:
        output_dir = TEMP_OUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    num_files = len(os.listdir(output_dir))
    filename = f"{num_files:05d}.mp4"
    output_path = os.path.join(output_dir, filename)

    if sound_path is None:
        temp_sound = os.path.join(TEMP_DIR, "sound.mp3")
        if os.path.exists(temp_sound):
            sound_path = temp_sound

    if frame_rate is None:
        frame_rate = 25  # Default frame rate for ffmpeg

    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(frame_rate),
        '-i', os.path.join(frames_dir, "%05d.jpg"),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    if sound_path is not None:
        command += [
            '-i', sound_path,
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            '-shortest'
        ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred while creating video from frames")
    return output_path


def get_frames_from_dir(vid_dir: str,
                        available_extensions: Optional[Union[List, str]] = None,
                        as_numpy: bool = False) -> List:
    """Get image file paths list from the dir"""
    if available_extensions is None:
        available_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]

    if isinstance(available_extensions, str):
        available_extensions = [available_extensions]

    frame_names = [
        p for p in os.listdir(vid_dir)
        if os.path.splitext(p)[-1] in available_extensions
    ]
    if not frame_names:
        return []
    frame_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

    frames = [os.path.join(vid_dir, name) for name in frame_names]
    if as_numpy:
        frames = [np.array(Image.open(frame)) for frame in frames]

    return frames


def clean_temp_dir(temp_dir: Optional[str] = None):
    """Removes media files from the directory."""
    if temp_dir is None:
        temp_dir = TEMP_DIR
        temp_out_dir = TEMP_OUT_DIR
    else:
        temp_out_dir = os.path.join(temp_dir, "out")

    clean_files_with_extension(temp_dir, SOUND_FILE_EXT)
    clean_files_with_extension(temp_dir, IMAGE_FILE_EXT)
    clean_files_with_extension(temp_out_dir, IMAGE_FILE_EXT)


def clean_files_with_extension(dir_path: str, extensions: List):
    """Remove files with the given extensions from the directory."""
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(tuple(extensions)):
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error while removing {file_path}: {str(e)}")

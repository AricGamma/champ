import os
from typing import Generator
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import imageio
from einops import rearrange
import torchvision.transforms as transforms


def save_videos_from_pil(pil_images, path, fps=24, crf=23):

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=24):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def resize_tensor_frames(video_tensor, new_size):
    B, C, video_length, H, W = video_tensor.shape
    # Reshape video tensor to combine batch and frame dimensions: (B*F, C, H, W)
    video_tensor_reshaped = video_tensor.reshape(-1, C, H, W)
    # Resize using interpolate
    resized_frames = F.interpolate(
        video_tensor_reshaped, size=new_size, mode="bilinear", align_corners=False
    )
    resized_video = resized_frames.reshape(B, C, video_length, new_size[0], new_size[1])

    return resized_video


def pil_list_to_tensor(image_list, size=None):
    to_tensor = transforms.ToTensor()
    if size is not None:
        tensor_list = [to_tensor(img.resize(size[::-1])) for img in image_list]
    else:
        tensor_list = [to_tensor(img) for img in image_list]
    stacked_tensor = torch.stack(tensor_list, dim=0)
    tensor = stacked_tensor.permute(1, 0, 2, 3)
    return tensor

def frames_to_video(images: Generator[Image.Image | str, None, None], output: str, fps=30.0):
    """image frames to video

    Args:
        images (list[Image.Image  |  str]): image frames
        output (str): output file path, must end with .mp4
        fps (int, optional): fps. Defaults to 30.

    Raises:
        ValueError: If images is empty
    """
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video: cv2.VideoWriter = None
    width = 0
    height = 0

    for img in images:
        if isinstance(img, str):
            img = cv2.imread(img)

        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # use first image size as determine video frame size
        if not width and not height:
            height, width, _ = img.shape

        if not video:
            video = cv2.VideoWriter(output, fourcc, fps, (width, height))

        video.write(img)

    video.release()

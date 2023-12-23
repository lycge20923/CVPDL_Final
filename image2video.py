import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image, ImageOps


def image_to_video(image_path, output_path):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    original_image = load_image(image_path)
    original_width, original_height = original_image.size

    if original_width / original_height > 1024/576:
        resized_width = int(original_width * 576 / original_height)
        resized_image = original_image.resize((resized_width, 576))
    
    else:
        resized_height = int(original_height * 1024 / original_width)
        resized_image = original_image.resize((1024, resized_height))

    image = ImageOps.fit(resized_image, (1024, 576), method=0, bleed=0.0, centering=(0.5, 0.5))

    generator = torch.manual_seed(23)
    frames = pipe(image, decode_chunk_size=8, generator=generator, num_frames=25, fps=6).frames[0]

    export_to_video(frames, output_path, fps=6)

if __name__ == '__main__':
    img_path = '/home/r12922169/course/test/CVPDL_Final/aaa.jpg'
    image_to_video(img_path, 'bbb.mp4')

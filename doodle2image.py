from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import Tuple


def doodle2image(in_dir, style_name, prompt, out_dir):
    style_list = [
        {
            "name": "(No style)",
            "prompt": "{prompt}",
            "negative_prompt": "",
        },
        {
            "name": "Cinematic",
            "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
            "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        },
        {
            "name": "3D Model",
            "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
            "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
        },
        {
            "name": "Anime",
            "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
            "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
        },
        {
            "name": "Digital Art",
            "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
            "negative_prompt": "photo, photorealistic, realism, ugly",
        },
        {
            "name": "Photographic",
            "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
        },
        {
            "name": "Pixel art",
            "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
            "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
        },
        {
            "name": "Fantasy art",
            "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
            "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
        },
        {
            "name": "Neonpunk",
            "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
            "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
        },
        {
            "name": "Manga",
            "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
            "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
        },
    ]
    styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
    STYLE_NAMES = list(styles.keys())
    DEFAULT_STYLE_NAME = "(No style)"
    def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + negative
    # load adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
            adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to(device)
    else:
        pipe = None
    MAX_SEED = np.iinfo(np.int32).max
    seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = load_image(in_dir)
    image = image.convert("RGB")
    image = TF.to_tensor(image) > 0.5
    image = TF.to_pil_image(image.to(torch.float32))
    prompt, negative_prompt = apply_style(style_name, prompt)
    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=30,
        generator=generator,
        guidance_scale=30,
        adapter_conditioning_scale=0.6,
        adapter_conditioning_factor=0.6
    ).images[0]
    gen_images.save(out_dir)
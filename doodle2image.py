from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.lineart import LineartDetector
import torch
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
    adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
    ).to("cuda")

    # load euler_a scheduler
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    image = load_image(in_dir)
    prompt, negative_prompt = apply_style(style_name, prompt)
    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=30,
        adapter_conditioning_scale=0.8,
        guidance_scale=7.5, 
    ).images[0]
    gen_images.save(out_dir)
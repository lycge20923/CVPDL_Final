from doodle2image import doodle2image
from image2video import image_to_video
import VideoToAudio
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doodle_path", type=str, help="path of your doodle")
    parser.add_argument("--style", type=str, choices=[
        "(No style)",
        "Cinematic",
        "3D Model",
        "Anime",
        "Digital Art",
        "Photographic",
        "Pixel art",
        "Fantasy art",
        "Neonpunk",
        "Manga"
    ], help="put your style in ''")
    parser.add_argument("--prompt", type=str, help="put your prompt in ''")
    args = parser.parse_args()

    doodle_path = args.doodle_path
    style = args.style
    prompt = args.prompt
    
    image_path = "image.jpg"
    video_path = "video.mp4"
    
    
    


    image = doodle2image(doodle_path, style, prompt, image_path)
    video = image_to_video(image_path, video_path)
    VideoToAudio.run(video_path)

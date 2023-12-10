import os
import sys
from doodle2image import doodle2image
from image2video import image_to_video
import VideoToAudio


if __name__ == '__main__':
    doodle_path = sys.argv[1]
    prompt = "a boy is plaing a geme under sun."
    image_path = "image.jpg"
    video_path = "video.mp4"
    
    
    


    image = doodle2image(doodle_path, 'Photographic', prompt, image_path)
    video = image_to_video(image_path, video_path)
    VideoToAudio.run(video_path)
